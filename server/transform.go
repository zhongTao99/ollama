package server

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/ollama/ollama/progress"
)

// type CommunityTransform interface {
// 	Transform() string
// }

// TransformFn is transform for a type of model.
type TransformFn func(ctx context.Context, mp ModelPath) (string, error)

func TransferringModelData(path string) (string, error) {
	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	status := "transferring model data"
	spinner := progress.NewSpinner(status)
	p.Add(status, spinner)
	defer p.Stop()

	bin, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer bin.Close()

	// Get file info to retrieve the size
	// fileInfo, err := bin.Stat()
	// if err != nil {
	// 	return "", err
	// }
	// fileSize := fileInfo.Size()

	hash := sha256.New()
	if _, err := io.Copy(hash, bin); err != nil {
		return "", err
	}

	if _, err := bin.Seek(0, io.SeekStart); err != nil {
		return "", err
	}

	// var pw progressWriter
	zeroStatus := "transferring model data 0%"
	spinner.SetMessage(zeroStatus)

	done := make(chan struct{})
	defer close(done)

	go func() {
		ticker := time.NewTicker(60 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				// spinner.SetMessage(fmt.Sprintf("transferring model data %d%%", int(100*pw.n.Load()/fileSize)))
				spinner.SetMessage(fmt.Sprintf("transferring model data ..."))
			case <-done:
				spinner.SetMessage("transferring model data 100%")
				return
			}
		}
	}()

	digest := fmt.Sprintf("sha256:%x", hash.Sum(nil))

	return digest, nil
}
