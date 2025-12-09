package main

import (
	"context"
	"fmt"
	"os"

	"dagger.io/dagger"
)

// create a background context and run the pipeline.
func main() {
	ctx := context.Background()
	if err := run(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "pipeline failed: %v\n", err)
		os.Exit(1)
	}
	return
}

// run orchestrates the pipeline steps using the Dagger Go SDK. It builds the repository image, then runs data -> train -> evaluate -> deploy inside that image, while capturing logs and exporting artifacts
func run(ctx context.Context) error {
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	//Build image from repo root and prepare a container with /app workspace.
	project := client.Host().Directory("..")
	image := project.DockerBuild()
	ctr := image.WithDirectory("/app", project).WithWorkdir("/app")
	current := ctr

	//Helper that runs a command in the current container state, prints stdout/stderr,
	run := func(name string, cmd []string, artifact string) (*dagger.Container, error) {
		fmt.Printf("--- Running step: %s ---\n", name)
		c := current.WithExec(cmd)

		// Sync first to execute
		_, execErr := c.Sync(ctx)

		// Get output after execution
		out, _ := c.Stdout(ctx)
		errout, _ := c.Stderr(ctx)

		// Always print output for debugging
		if out != "" {
			fmt.Printf("[%s] STDOUT:\n%s\n", name, out)
		}
		if errout != "" {
			fmt.Printf("[%s] STDERR:\n%s\n", name, errout)
		}

		if execErr != nil {
			if artifact != "" {
				// Print artifact contents on failure to aid debugging.
				cat := []string{"sh", "-c", fmt.Sprintf("if [ -f %s ]; then cat %s; else echo 'no %s'; fi", artifact, artifact, artifact)}
				cc := c.WithExec(cat)
				clog, _ := cc.Stdout(ctx)
				if clog != "" {
					fmt.Printf("[%s] ARTIFACT %s:\n%s\n", name, artifact, clog)
				}
			}
			return nil, fmt.Errorf("step %s failed: %w", name, execErr)
		}
		//Persist container state for the next step.
		current = c
		return c, nil
	}

	//print the python version inside the built image.
	if _, err := run("python-version", []string{"python", "--version"}, ""); err != nil {
		return err
	}

	//Data preprocessing: run `src/data/data.py`. If it raises, the wrapper
	//writes the traceback to `artifacts/data_py_error.log` inside the container.
	dataCmd := []string{"python", "-u", "-c", "import runpy,traceback,sys;\ntry:\n runpy.run_path('src/data/data.py', run_name='__main__')\nexcept Exception:\n tb=traceback.format_exc(); open('artifacts/data_py_error.log','w').write(tb); traceback.print_exc(); sys.exit(1)\nelse:\n print('data.py completed')"}
	if _, err := run("data", dataCmd, "artifacts/data_py_error.log"); err != nil {
		return err
	}

	//Copy the preprocessed train file into ./artifacts so train.py can read it.
	prep := []string{"sh", "-c", "mkdir -p artifacts && cp -v src/data/artifacts/train_data_gold.csv artifacts/ || true; ls -la artifacts"}
	if _, err := run("prepare-data", prep, ""); err != nil {
		return err
	}

	//Train: run the training script and capture logs to `artifacts/train_run.log`.
	//run a short Python wrapper so stdout/stderr and any traceback are reliably written to the log file inside the container.
	trainCmd := []string{"python", "-u", "-c", "import runpy,traceback,sys; f=open('artifacts/train_run.log','w'); sys.stdout=f; sys.stderr=f; \ntry:\n runpy.run_path('src/models/train.py', run_name='__main__')\nexcept Exception:\n tb=traceback.format_exc(); f.write('\\n--- EXCEPTION ---\\n'); f.write(tb); f.flush(); raise\nelse:\n f.write('\\ntrain.py completed\\n'); f.flush()\nfinally:\n f.close()"}
	if _, err := run("train", trainCmd, "artifacts/train_run.log"); err != nil {
		return err
	}

	//Export the container `artifacts/` directory to the host repo `../artifacts`.
	artDir := current.Directory("artifacts")
	if _, err := artDir.Export(ctx, "../artifacts"); err != nil {
		return fmt.Errorf("export failed: %w", err)
	}

	//Evaluate and deploy steps follow the same pattern: run script, write any tracebacks into artifact logs, and print completion message.
	evalCmd := []string{"python", "-u", "-c", "import runpy,traceback,sys;\ntry:\n runpy.run_path('src/models/evaluate.py', run_name='__main__')\nexcept Exception:\n tb=traceback.format_exc(); open('artifacts/evaluate_py_error.log','w').write(tb); traceback.print_exc(); sys.exit(1)\nelse:\n print('evaluate.py completed')"}
	if _, err := run("evaluate", evalCmd, "artifacts/evaluate_py_error.log"); err != nil {
		return err
	}

	deployCmd := []string{"python", "-u", "-c", "import runpy,traceback,sys;\ntry:\n runpy.run_path('src/models/Deploy.py', run_name='__main__')\nexcept Exception:\n tb=traceback.format_exc(); open('artifacts/deploy_py_error.log','w').write(tb); traceback.print_exc(); sys.exit(1)\nelse:\n print('Deploy.py completed')"}
	if _, err := run("deploy", deployCmd, "artifacts/deploy_py_error.log"); err != nil {
		return err
	}

	return nil
}
