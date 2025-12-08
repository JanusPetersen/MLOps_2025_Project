package main

import (
    "context"
    "fmt"

    "dagger.io/dagger"
)

func main() {
    // Create a shared context
    ctx := context.Background()

    // Run the stages of the pipeline
    if err := Build(ctx); err != nil {
        fmt.Println("Error:", err)
        panic(err)
    }
}

func Build(ctx context.Context) error {
    // Initialize Dagger client
    client, err := dagger.Connect(ctx)
    if err != nil {
        return err
    }
    defer client.Close()

    python := client.Container().
        Build(client.Host().Directory(".")).
        WithWorkdir("/app")


    python = python.WithExec([]string{"python", "--version"})

    python = python.WithExec([]string{"python", "data/data.py"})
    python = python.WithExec([]string{"python", "models/train.py"})
    python = python.WithExec([]string{"python", "models/evaluate.py"})
    python = python.WithExec([]string{"python", "models/Deploy.py"})

    _, err = python.
        Directory("output").
        Export(ctx, "output")
    if err != nil {
        return err
    }

    return nil
}