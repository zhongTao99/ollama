package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

const (
	DefaultFactoryRegistry = "huggingface.co"
	DefaultQuantify        = "Q4_0"
	DefaultBranch          = "main"
	DefaultFormat          = "gguf"
)

var (
	// Used to validate if the registry is supportable
	SupportCommunityRegistry        = []string{"huggingface.co", "modelers.cn"}
	SupportCommunityRegistrySubPath = []string{"%s/resolve/main/%s", "/coderepo/web/v1/file/%s/main/media/%s"}
)

const (
	AgreementModelURL      = "https://%s/%s"
	AgreementDatasetURL    = "https://%s/datasets/%s"
	RawModelFileURL        = "https://%s/%s/raw/%s/%s"
	RawDatasetFileURL      = "https://%s/datasets/%s/raw/%s/%s"
	LfsModelResolverURL    = "https://%s/%s/resolve/%s/%s"
	LfsDatasetResolverURL  = "https://%s/datasets/%s/resolve/%s/%s"
	JsonModelsFileTreeURL  = "https://%s/api/models/%s/tree/%s/%s"
	JsonDatasetFileTreeURL = "https://%s/api/datasets/%s/tree/%s/%s"
)

// type CommunityDownload interface {
// 	Download() string
// }

// DownloadFn is download for a type of model.
type DownloadFn func(ctx context.Context, mp ModelPath) (string, error)

type CommunityModel struct {
	modelType string
	// model info
	downloadFn  DownloadFn
	transformFn TransformFn
}

func IsSupportCommunityRegistry(registry string) bool {
	for _, item := range SupportCommunityRegistry {
		if registry == item {
			return true
		}
	}
	return false
}

func GetSupportCommunityRegistrySubPath(registry string) string {
	for i, item := range SupportCommunityRegistry {
		if registry == item {
			return SupportCommunityRegistrySubPath[i]
		}
	}
	return ""
}

func createModelBlob(digest string, file string) error {
	bin, err := os.Open(file)
	if err != nil {
		return err
	}
	defer bin.Close()

	if ib, ok := intermediateBlobs[digest]; ok {
		p, err := GetBlobsPath(ib)
		if err != nil {
			return err
		}

		if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
			slog.Info("evicting intermediate blob which no longer exists", "digest", ib)
			delete(intermediateBlobs, digest)
		} else if err != nil {
			return err
		} else {
			return nil
		}
	}

	path, err := GetBlobsPath(digest)
	if err != nil {
		return err
	}

	_, err = os.Stat(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop
	case err != nil:
		return err
	default:
		return nil
	}

	layer, err := NewLayer(bin, "")
	if err != nil {
		return nil
	}

	if layer.Digest != digest {
		slog.Info("digest mismatch, expected %q, got %q", digest, layer.Digest)
		return fmt.Errorf("digest mismatch, expected %q, got %q", digest, layer.Digest)
	}
	return nil
}

func PullModelFromCommunityRegistry(ctx context.Context, mp ModelPath, regOpts *registryOptions, fn func(api.ProgressResponse)) error {
	slog.Info(fmt.Sprintf("yyyyyyyyyyyyyyyyyyyyy mp.Tag %s", mp.Tag))
	mp.subPath = GetSupportCommunityRegistrySubPath(mp.Registry)
	if mp.subPath == "" {
		return fmt.Errorf("Failed to get community registry subPath")
	}

	count := strings.Count(mp.Tag, ":")
	if count > 2 {
		return fmt.Errorf("Invalid model name")
	}

	slog.Info(fmt.Sprintf("yyyyyyyyyyyyyyyyyyyyy count %d", count))

	var modelFiles []string
	var digests []string
	modelFiles = strings.Split(mp.Tag, ":")

	for _, modelFile := range modelFiles {
		// 在这里处理每个 modelFile
		mp.fileName = modelFile + ".gguf"

		// download
		// fn(api.ProgressResponse{Status: "pulling community model"})
		tmpFile, err := downloadCommunityModel(ctx, downloadOpts{
			mp:      mp,
			digest:  "",
			regOpts: regOpts,
			fn:      fn,
			name:    mp.fileName,
		})
		defer os.Remove(tmpFile)
		if err != nil {
			return err
		}

		// todo: cleanup?

		slog.Info(fmt.Sprintf("yyyyyyyyyyyyyyyyyyyyy tmpFile is %s", tmpFile))

		// transform

		digest, err := TransferringModelData(tmpFile)
		if err != nil {
			return err
		}

		err = createModelBlob(digest, tmpFile)
		if err != nil {
			return err
		}
		digests = append(digests, digest)
	}

	name := model.ParseName(mp.Repository + ":" + mp.Tag)
	if !name.IsValid() {
		return fmt.Errorf("%s", errtypes.InvalidModelNameErrMsg)
	}
	slog.Info(fmt.Sprintf("yyyyyyyyyyyyyyyyyyyyy name is %s, digests: %s", name, digests))

	if err := checkNameExists(name); err != nil {
		return err
	}
	if err := CreateCommunityModel(ctx, name, digests, fn); err != nil {
		slog.Info(fmt.Sprintf("yyyyyyyyyyyyyyyyyyyyy err:%s", err))
		return err
	}

	// os.Remove(tmpFile)

	return nil
}

func CreateCommunityModel(ctx context.Context, name model.Name, digestArr []string, fn func(resp api.ProgressResponse)) (err error) {
	config := ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
		RootFS: RootFS{
			Type: "layers",
		},
	}

	// name:registry.ollama.ai/library/Llama3:8b-q4-k-m modelFileDir: . quantization: "
	slog.Info(fmt.Sprintf("yyyyyyyyyyyyyyyyyyyyy CreateModel name:%s digest: %s ", name, digestArr))

	var messages []*api.Message
	parameters := make(map[string]any)

	var layers []Layer
	var baseLayers []*layerGGML

	for _, digest := range digestArr {
		if ib, ok := intermediateBlobs[digest]; ok {
			p, err := GetBlobsPath(ib)
			if err != nil {
				return err
			}

			if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
				// pass
			} else if err != nil {
				return err
			} else {
				fn(api.ProgressResponse{Status: fmt.Sprintf("using cached layer %s", ib)})
				digest = ib
			}
		}

		blobpath, err := GetBlobsPath(digest)
		if err != nil {
			return err
		}

		slog.Info(fmt.Sprintf("yyyyyyyyyyyyyyyyyyyyy blobpath:%s ", blobpath))

		blob, err := os.Open(blobpath)
		if err != nil {
			return err
		}
		defer blob.Close()

		var command string
		baseLayers, err = parseFromFile(ctx, command, baseLayers, blob, digest, fn)
		if err != nil {
			return err
		}
	}

	for _, baseLayer := range baseLayers {
		if baseLayer.GGML != nil {
			config.ModelFormat = cmp.Or(config.ModelFormat, baseLayer.GGML.Name())
			config.ModelFamily = cmp.Or(config.ModelFamily, baseLayer.GGML.KV().Architecture())
			config.ModelType = cmp.Or(config.ModelType, format.HumanNumber(baseLayer.GGML.KV().ParameterCount()))
			config.FileType = cmp.Or(config.FileType, baseLayer.GGML.KV().FileType().String())
			config.ModelFamilies = append(config.ModelFamilies, baseLayer.GGML.KV().Architecture())
		}

		layers = append(layers, baseLayer.Layer)
	}

	var err2 error
	layers = slices.DeleteFunc(layers, func(layer Layer) bool {
		switch layer.MediaType {
		case "application/vnd.ollama.image.message":
			// if there are new messages, remove the inherited ones
			if len(messages) > 0 {
				return true
			}

			return false
		case "application/vnd.ollama.image.params":
			// merge inherited parameters with new ones
			r, err := layer.Open()
			if err != nil {
				err2 = err
				return false
			}
			defer r.Close()

			var ps map[string]any
			if err := json.NewDecoder(r).Decode(&ps); err != nil {
				err2 = err
				return false
			}

			for k, v := range ps {
				if _, ok := parameters[k]; !ok {
					parameters[k] = v
				}
			}

			return true
		default:
			return false
		}
	})

	if err2 != nil {
		return err2
	}

	if len(messages) > 0 {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(messages); err != nil {
			return err
		}

		layer, err := NewLayer(&b, "application/vnd.ollama.image.messages")
		if err != nil {
			return err
		}

		layers = append(layers, layer)
	}

	if len(parameters) > 0 {
		var b bytes.Buffer
		if err := json.NewEncoder(&b).Encode(parameters); err != nil {
			return err
		}

		layer, err := NewLayer(&b, "application/vnd.ollama.image.params")
		if err != nil {
			return err
		}

		layers = append(layers, layer)
	}

	digests := make([]string, len(layers))
	for i, layer := range layers {
		digests[i] = layer.Digest
	}

	config.RootFS.DiffIDs = digests

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(config); err != nil {
		return err
	}

	configLayer, err := NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return err
	}

	for _, layer := range append(layers, configLayer) {
		if layer.status != "" {
			fn(api.ProgressResponse{Status: layer.status})
		}
	}

	old, _ := ParseNamedManifest(name)

	fn(api.ProgressResponse{Status: "writing manifest"})
	if err := WriteManifest(name, configLayer, layers); err != nil {
		return err
	}

	if !envconfig.NoPrune() && old != nil {
		if err := old.RemoveLayers(); err != nil {
			return err
		}
	}

	fn(api.ProgressResponse{Status: "success"})
	return nil
}

func GetModelTmpPath() (string, error) {
	randName := fmt.Sprintf("model-%d", time.Now().UnixNano())

	tmpPath := fmt.Sprintf("/tmp/%s", randName)
	return tmpPath, nil
}
