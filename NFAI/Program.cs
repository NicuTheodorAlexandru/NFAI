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
