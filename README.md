# NFAI

> **Note:** NFAI is in a very early stage of development. Expect rapid changes, missing features, and breaking changes.

NFAI is a native .NET inference engine for [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) models, focused on efficient inference for Llama 3.2 models in FP16 and FP32 precision. It leverages Vulkan compute shaders for high performance on modern GPUs.

## Features

- **Native .NET**: Written in C#, no Python dependencies.
- **GGUF Support**: Loads and runs GGUF models (currently Llama 3.2 only, in theory should work with any Llama 3.X).
- **Vulkan Backend**: Uses Vulkan compute shaders for fast inference.
- **Precision**: Supports both FP16 and FP32 weights.
- **Cross-vendor GPU**: Tested on Windows with both AMD and NVIDIA GPUs.

## Requirements

- [.NET 9.0+](https://dotnet.microsoft.com/en-us/download)
- [glslangValidator](https://github.com/KhronosGroup/glslang) (must be in your PATH, I personally use the lunarg vulkan sdk for this)
- Vulkan-compatible GPU and drivers (AMD or NVIDIA)
- Windows OS (tested)

## Usage

1. **Install dependencies**  
   - Ensure Vulkan drivers are installed for your GPU.
   - Download and install `glslangValidator` and add it to your system PATH.

2. **Prepare a GGUF model**  
   - Obtain a Llama 3.2 model in GGUF format (FP16 or FP32).

3. **Set the GGUF path**  
   - Set the `GGUF_PATH` environment variable to point to your model file.

4. **Run the program**  

   ```powershell
   dotnet run --project NFAI
   ```

   You will be prompted for input in the console.

## Example

Below is a minimal example from `Program.cs`:

```csharp
// parse the GGUF file
using Microsoft.Extensions.AI;
using NFAI.GGUF;

var path = Environment.GetEnvironmentVariable("GGUF_PATH") ?? throw new ArgumentNullException("GGUF_PATH", "GGUF_PATH environment variable is not set.");
var parser = new Parser();
var model = parser.Parse(path);
Console.WriteLine("Enter 'quit' to quit: ");

var input = string.Empty;
while (input != "quit")
{
    Console.Write("User: ");
    input = Console.ReadLine() ?? string.Empty;
    if (input == "quit") break;
    Console.Write("Assistant: ");
    await foreach(var part in model.GetStreamingResponseAsync(input, default))
    {
        Console.Write(part);
    }
    Console.WriteLine();
}
```

## Notes

- Only Llama 3.2 GGUF models are supported at this time.
- Model quantization formats other than FP16/FP32 are not yet supported.
- Performance and compatibility may vary depending on your GPU and drivers.

## Roadmap

- Add support for more GGUF model types and quantizations.
- Improve tokenizer compatibility.
- Add Linux support.

## License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0). See the [LICENSE](LICENSE) file for details.

---

**Contributions and issues welcome!**
