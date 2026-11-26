class Llmkit < Formula
  desc "Command-line toolkit for working with Large Language Models"
  homepage "https://github.com/aktagon/llmkit-go"
  # The url, version, and sha256 change with each release.
  url "https://github.com/aktagon/llmkit-go/archive/refs/tags/go-v4.1.2.tar.gz"
  version "go-v4.1.2"
  sha256 "fabb7272c3932164b0a297e93ae7965e9ee67c0dcfa6216ff8009821367a475c"
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
