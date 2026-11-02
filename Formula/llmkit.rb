class Llmkit < Formula
  desc "Command-line toolkit for working with Large Language Models"
  homepage "https://github.com/aktagon/llmkit-go"
  # NOTE: The url, version, and sha256 are updated by the github action (.github/workflows/release.yml) automatically
  url "https://github.com/aktagon/llmkit-go/archive/refs/tags/go-v3.1.0.tar.gz"
  version "go-v3.1.0"
  sha256 "36eaac5f49d828cfa7d9a9e4e71db48460a2a2ae42a23d3c93d807f37aa4362e"
  license "MIT"

  depends_on "go" => :build

  def install
    system "go", "build", *std_go_args(ldflags: "-s -w"), "-o", bin/"llmkit", "./cmd/llmkit"
  end

  def caveats
    <<~EOS
      llmkit provides unified access to multiple LLM providers.

      Set your API keys as environment variables:
        export OPENAI_API_KEY="your-openai-key"
        export ANTHROPIC_API_KEY="your-anthropic-key"
        export GOOGLE_API_KEY="your-google-key"

      You can add these to your shell profile (~/.zshrc, ~/.bashrc, etc.)
      to make them permanent.

      Use 'llmkit --help' to see available commands and options.
    EOS
  end

  test do
    # Test that the binary was installed and can display help
    assert_match "llmkit", shell_output("#{bin}/llmkit --help 2>&1", 1)
  end
end
