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

	//Helper that runs a command in the current container state
	run := func(name string, cmd []string) (*dagger.Container, error) {
		c := current.WithExec(cmd)
		if _, err := c.Sync(ctx); err != nil {
			return nil, fmt.Errorf("step %s failed: %w", name, err)
		}
		//Persist container state for the next step
		current = c
		return c, nil
	}

	//Data preprocessing: run `src/data/data.py`
	dataCmd := []string{"python", "-u", "src/data/data.py"}
	if _, err := run("data", dataCmd); err != nil {
		return err
	}

	//Copy the preprocessed train file into ./artifacts so train.py can read it
	prep := []string{"sh", "-c", "mkdir -p artifacts && cp src/data/artifacts/train_data_gold.csv artifacts/"}
	if _, err := run("prepare-data", prep); err != nil {
		return err
	}

	//Train: run the training script
	trainCmd := []string{"python", "-u", "src/models/train.py"}
	if _, err := run("train", trainCmd); err != nil {
		return err
	}

	//Export the container `artifacts/` directory to the host repo `../artifacts`.
	artDir := current.Directory("artifacts")
	if _, err := artDir.Export(ctx, "../artifacts"); err != nil {
		return fmt.Errorf("export failed: %w", err)
	}

	//Evaluate: run model evaluation
	evalCmd := []string{"python", "-u", "src/models/evaluate.py"}
	if _, err := run("evaluate", evalCmd); err != nil {
		return err
	}

	//Deploy: register model
	deployCmd := []string{"python", "-u", "src/models/Deploy.py"}
	if _, err := run("deploy", deployCmd); err != nil {
		return err
	}

	return nil
}
