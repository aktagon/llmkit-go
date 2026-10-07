class Llmkit < Formula
  desc "Command-line toolkit for working with Large Language Models"
  homepage "https://github.com/aktagon/llmkit-go"
  # NOTE: The url, version, and sha256 are updated by the github action (.github/workflows/release.yml) automatically
  url "https://github.com/aktagon/llmkit-go/archive/refs/tags/go-v2.2.2.tar.gz"
  version "go-v2.2.2"
  sha256 "b9f622f972bdf13f29c697837d67a7e0067222027c46e1f1c55fddfb29d9d9d2"
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
