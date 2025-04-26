using Microsoft.Extensions.AI;
using Microsoft.Extensions.Hosting;

namespace NFAI;

public class ChatService(IChatClient chatClient) : BackgroundService
{
    protected async override Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var input = string.Empty;
        while (input != "quit")
        {
            Console.Write("User: ");
            input = Console.ReadLine() ?? string.Empty;
            if (input == "quit") break;
            Console.Write("Assistant: ");
            await foreach(var part in chatClient.GetStreamingResponseAsync(input, default))
            {
                Console.Write(part);
            }
            Console.WriteLine();
        }
    }
}