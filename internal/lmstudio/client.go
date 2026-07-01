// Package lmstudio is a tiny client for talking to LM Studio's local server.
//
// LM Studio exposes an "OpenAI-compatible" HTTP API. That means it speaks the
// same request/response shapes as OpenAI's API, but runs entirely on your
// machine (default address: http://localhost:1234/v1). We only need two
// endpoints:
//
//   POST /v1/embeddings      -> turn text into a vector ([]float32)
//   POST /v1/chat/completions -> ask the chat model a question, get text back
//
// Everything here is plain net/http + encoding/json from Go's standard
// library. No third-party dependencies. Read it top to bottom and you'll see
// exactly what bytes go over the wire.
package lmstudio

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client holds the settings for reaching your LM Studio server.
type Client struct {
	BaseURL    string // e.g. "http://localhost:1234/v1"
	EmbedModel string // the embedding model loaded in LM Studio
	ChatModel  string // the chat model loaded in LM Studio
	httpClient *http.Client
}

// New builds a Client. We give it a generous timeout because local models can
// take a few seconds to warm up on the first request.
func New(baseURL, embedModel, chatModel string) *Client {
	return &Client{
		BaseURL:    baseURL,
		EmbedModel: embedModel,
		ChatModel:  chatModel,
		httpClient: &http.Client{Timeout: 120 * time.Second},
	}
}

// ---- Embeddings -----------------------------------------------------------

// embedRequest is the JSON body we POST to /v1/embeddings.
type embedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

// embedResponse is the shape LM Studio sends back. We only pull out the vector.
type embedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed converts a single piece of text into a vector of numbers. Similar text
// produces similar vectors — that's the whole magic that makes search work.
func (client *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(embedRequest{Model: client.EmbedModel, Input: text})
	if err != nil {
		return nil, err
	}

	response, err := client.post(ctx, "/embeddings", body)
	if err != nil {
		return nil, err
	}

	var parsed embedResponse
	if err := json.Unmarshal(response, &parsed); err != nil {
		return nil, fmt.Errorf("decoding embedding response: %w", err)
	}
	if len(parsed.Data) == 0 || len(parsed.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("LM Studio returned an empty embedding (is an embedding model loaded?)")
	}
	return parsed.Data[0].Embedding, nil
}

// ---- Chat -----------------------------------------------------------------

// Message is one turn in a chat conversation. Role is "system", "user", or
// "assistant".
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
}

type chatResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
}

// Chat sends a list of messages to the chat model and returns its reply text.
// A low temperature keeps answers focused and factual — good for RAG.
func (client *Client) Chat(ctx context.Context, messages []Message) (string, error) {
	body, err := json.Marshal(chatRequest{
		Model:       client.ChatModel,
		Messages:    messages,
		Temperature: 0.2,
	})
	if err != nil {
		return "", err
	}

	response, err := client.post(ctx, "/chat/completions", body)
	if err != nil {
		return "", err
	}

	var parsed chatResponse
	if err := json.Unmarshal(response, &parsed); err != nil {
		return "", fmt.Errorf("decoding chat response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("LM Studio returned no chat choices (is a chat model loaded?)")
	}
	return parsed.Choices[0].Message.Content, nil
}

// ---- shared plumbing ------------------------------------------------------

// post sends a JSON body to BaseURL+path and returns the raw response bytes.
// It turns connection failures into a friendly, actionable message — because
// "connection refused" is the #1 thing you'll hit while learning.
func (client *Client) post(ctx context.Context, path string, body []byte) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, client.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := client.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf(
			"could not reach LM Studio at %s\n"+
				"  -> Is LM Studio running with the local server started? (Developer tab -> Start Server)\n"+
				"  -> underlying error: %w", client.BaseURL, err)
	}
	defer response.Body.Close()

	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("LM Studio returned HTTP %d: %s", response.StatusCode, string(responseBody))
	}
	return responseBody, nil
}
