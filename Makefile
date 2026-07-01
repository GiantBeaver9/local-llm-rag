# Build order matters: the Rust static library must exist before `go build`,
# because cgo links it into the Go binary. These targets encode that order so
# you never have to remember it.

RUST_DIR := rust/vindex
RUST_LIB := $(RUST_DIR)/target/release/libvindex.a

.PHONY: all build rust go test test-rust test-go clean

# Default: build everything, Rust first.
all: build

build: rust go

# Compile the Rust vector index into a static archive.
rust:
	cd $(RUST_DIR) && cargo build --release

# Build the Go binary. Depends on the Rust archive existing.
go: $(RUST_LIB)
	go build -o rag ./cmd/rag

# Rebuild the archive if any Rust source changed.
$(RUST_LIB): $(RUST_DIR)/src/lib.rs $(RUST_DIR)/Cargo.toml
	cd $(RUST_DIR) && cargo build --release

# Run every test in the project (Rust unit tests + Go tests incl. the cgo bridge).
test: test-rust test-go

test-rust:
	cd $(RUST_DIR) && cargo test --release

test-go: $(RUST_LIB)
	go test ./...

clean:
	cd $(RUST_DIR) && cargo clean
	rm -f rag store.json
