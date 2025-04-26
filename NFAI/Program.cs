using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using NFAI;
using NFAI.GGUF;
using NFAI.Models;
using NFAI.Models.Llama3;

var builder = Host.CreateDefaultBuilder(args);

builder.ConfigureServices((context, services) =>
{
    services.AddLogging();
    services.Configure<ModelOptions>(context.Configuration.GetSection("ModelOptions"));
    services.AddSingleton<AbstractModelFactory, LlamaModelFactory>();
    services.AddSingleton<Parser>();
    services.AddSingleton(sp => 
    {
        var options = sp.GetRequiredService<IOptions<ModelOptions>>();
        var parser = sp.GetRequiredService<Parser>();
        return parser.Parse(options.Value);
    });
    services.AddSingleton<IChatClient, GenericChatClient>();
    services.AddHostedService<ChatService>();
});

var host = builder.Build();
await host.RunAsync();