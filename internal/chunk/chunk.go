// Package chunk splits long text into smaller overlapping pieces ("chunks").
//
// Why chunk at all? Two reasons:
//   1. Embedding models have a size limit, and they represent a small focused
//      passage far better than a whole book squeezed into one vector.
//   2. When we answer a question we only want to feed the model the few
//      relevant passages, not the entire document.
//
// We split on word boundaries and add a little overlap between neighbouring
// chunks so a sentence that straddles a boundary isn't lost.
package chunk

import "strings"

// Chunk is one piece of a source document.
type Chunk struct {
	Source string // which file it came from
	Index  int    // its position within that file (0, 1, 2, ...)
	Text   string // the actual text
}

// Split breaks text into chunks of roughly chunkWords words each, with
// overlapWords words shared between consecutive chunks.
//
// Word count is a simple, predictable proxy for size. It's not token-exact,
// but it's easy to reason about while you're learning — and good enough.
func Split(source, text string, chunkWords, overlapWords int) []Chunk {
	// Guard against silly inputs so callers can't create an infinite loop.
	if chunkWords <= 0 {
		chunkWords = 200
	}
	if overlapWords < 0 || overlapWords >= chunkWords {
		overlapWords = chunkWords / 5 // default: 20% overlap
	}

	words := strings.Fields(text) // splits on any whitespace, drops empties
	if len(words) == 0 {
		return nil
	}

	var chunks []Chunk
	step := chunkWords - overlapWords // how far we advance each iteration

	for start, i := 0, 0; start < len(words); start += step {
		end := start + chunkWords
		if end > len(words) {
			end = len(words)
		}

		piece := strings.Join(words[start:end], " ")
		chunks = append(chunks, Chunk{Source: source, Index: i, Text: piece})
		i++

		if end == len(words) {
			break // we've consumed everything
		}
	}
	return chunks
}
