using Silk.NET.Vulkan;
using Silk.NET.Vulkan.Extensions.EXT;

namespace NFAI;

public unsafe static class VulkanHelper
{
    private static DebugUtilsMessengerEXT debugMessenger;
    private static ExtDebugUtils debugUtils;

    public static unsafe Instance CreateVulkanInstance(Vk vk)
    {
        // 1) Setup an array of validation layers
        string[] desiredLayers = new[]
        {
            "VK_LAYER_KHRONOS_validation" // the main meta-layer
        };

            // 2) Setup an array of extensions
            //    We need the "VK_EXT_debug_utils" extension for debug logging.
        string[] desiredExtensions = new[]
        {
            ExtDebugUtils.ExtensionName, // from Silk.NET
            //"VK_KHR_8bit_storage"
        };

        // 3) Application info (optional but good practice)
        ApplicationInfo appInfo = new()
        {
            SType = StructureType.ApplicationInfo,
            PApplicationName = (byte*)Silk.NET.Core.Native.SilkMarshal.StringToPtr("MyVulkanApp"),
            ApiVersion = Vk.Version13
        };

        // 4) Create info for instance
        InstanceCreateInfo createInfo = new()
        {
            SType = StructureType.InstanceCreateInfo,
            PApplicationInfo = &appInfo
        };

        // 5) Fill in enabled layers
        var layerPtrs = new IntPtr[desiredLayers.Length];
        for (int i = 0; i < desiredLayers.Length; i++)
        {
            layerPtrs[i] = Silk.NET.Core.Native.SilkMarshal.StringToPtr(desiredLayers[i]);
        }
        createInfo.EnabledLayerCount = (uint)desiredLayers.Length;
        fixed (IntPtr* layerNamesPtr = layerPtrs)
        {
            createInfo.PpEnabledLayerNames = (byte**)layerNamesPtr;
        }

        // 6) Fill in enabled extensions
        var extensionPtrs = new IntPtr[desiredExtensions.Length];
        for (int i = 0; i < desiredExtensions.Length; i++)
        {
            extensionPtrs[i] = Silk.NET.Core.Native.SilkMarshal.StringToPtr(desiredExtensions[i]);
        }
        createInfo.EnabledExtensionCount = (uint)desiredExtensions.Length;
        fixed (IntPtr* extensionNamesPtr = extensionPtrs)
        {
            createInfo.PpEnabledExtensionNames = (byte**)extensionNamesPtr;
        }

        // 7) Create the instance
        Instance instance;
        var result = vk.CreateInstance(&createInfo, null, out instance);
        if (result != Result.Success)
        {
            throw new Exception("Failed to create Vulkan instance with validation.");
        }
        SetupDebugMessenger(vk, instance);

        // 8) Free the allocated strings (to avoid memory leak)
        for (int i = 0; i < desiredLayers.Length; i++)
        {
            Silk.NET.Core.Native.SilkMarshal.FreeString((nint)layerPtrs[i]);
        }
        for (int i = 0; i < desiredExtensions.Length; i++)
        {
            Silk.NET.Core.Native.SilkMarshal.FreeString((nint)extensionPtrs[i]);
        }

        return instance;
    }

    public static unsafe void SetupDebugMessenger(Vk vk, Instance instance)
    {
        // We must load the debug utils extension commands
        debugUtils = new ExtDebugUtils(vk.Context);

        DebugUtilsMessengerCreateInfoEXT createInfo = new DebugUtilsMessengerCreateInfoEXT
        {
            SType = StructureType.DebugUtilsMessengerCreateInfoExt,
            MessageSeverity =
                DebugUtilsMessageSeverityFlagsEXT.VerboseBitExt |
                DebugUtilsMessageSeverityFlagsEXT.InfoBitExt |
                DebugUtilsMessageSeverityFlagsEXT.WarningBitExt |
                DebugUtilsMessageSeverityFlagsEXT.ErrorBitExt,
            MessageType =
                DebugUtilsMessageTypeFlagsEXT.GeneralBitExt |
                DebugUtilsMessageTypeFlagsEXT.PerformanceBitExt |
                DebugUtilsMessageTypeFlagsEXT.ValidationBitExt,
            PfnUserCallback = new PfnDebugUtilsMessengerCallbackEXT(DebugCallback)
            // 'DebugCallback' is a static method
        };

        if (debugUtils.CreateDebugUtilsMessenger(instance, &createInfo, null, out debugMessenger) != Result.Success)
        {
            throw new Exception("Failed to create debug messenger!");
        }
    }

    // Our callback function
    private static uint DebugCallback(
        DebugUtilsMessageSeverityFlagsEXT messageSeverity,
        DebugUtilsMessageTypeFlagsEXT messageTypes,
        DebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        // Print out the message
        string message = Silk.NET.Core.Native.SilkMarshal.PtrToString((nint)pCallbackData->PMessage);
        Console.WriteLine($"[Validation Layer] {message}");

        if ((messageSeverity & DebugUtilsMessageSeverityFlagsEXT.ErrorBitExt) != 0)
        {
            // If it's an error, we can abort the program
            Console.WriteLine("Fatal error in Vulkan layer. Aborting.");
            Environment.Exit(1);
        }

        return Vk.False; // VK_FALSE to indicate that layer should not abort
    }

    public static PhysicalDevice PickPhysicalDevice(Vk vk, Instance instance)
    {
        uint deviceCount = 0;
        vk.EnumeratePhysicalDevices(instance, &deviceCount, null);
        if (deviceCount == 0)
            throw new Exception("No physical devices with Vulkan support found.");

        var physDevices = new PhysicalDevice[deviceCount];
        fixed (PhysicalDevice* ptr = physDevices)
        {
            vk.EnumeratePhysicalDevices(instance, &deviceCount, ptr);
        }

        // Just pick the last device for demo
        return physDevices[deviceCount - 1];
    }

    public static Device CreateLogicalDevice(Vk vk, PhysicalDevice physicalDevice)
    {
        // You need to find a queue family that supports compute
        // For demo, we'll assume it's family index 0 or pick properly
        uint queueFamilyIndex = 0; // <= find in real code
        float queuePriority = 1.0f;

        DeviceQueueCreateInfo queueCreateInfo = new()
        {
            SType = StructureType.DeviceQueueCreateInfo,
            QueueFamilyIndex = queueFamilyIndex,
            QueueCount = 1,
        };

        queueCreateInfo.PQueuePriorities = &queuePriority;

        // Check for 8-bit storage extension support
        uint extensionCount = 0;
        vk.EnumerateDeviceExtensionProperties(physicalDevice, (byte*)null, &extensionCount, null);
        
        ExtensionProperties[] availableExtensions = new ExtensionProperties[extensionCount];
        fixed (ExtensionProperties* extensionsPtr = availableExtensions)
        {
            vk.EnumerateDeviceExtensionProperties(physicalDevice, (byte*)null, &extensionCount, extensionsPtr);
        }
        
        for (int i = 0; i < extensionCount; i++)
        {
            string extName;
            fixed (byte* namePtr = availableExtensions[i].ExtensionName)
            {
                extName = Silk.NET.Core.Native.SilkMarshal.PtrToString((nint)namePtr);
            }
        }
        
        // Enable required extensions
        string[] deviceExtensions = [ "VK_KHR_storage_buffer_storage_class", "VK_EXT_shader_atomic_float" ];
        // Set up extension name pointers
        var extensionPtrs = new IntPtr[deviceExtensions.Length];
        for (int i = 0; i < deviceExtensions.Length; i++)
        {
            extensionPtrs[i] = Silk.NET.Core.Native.SilkMarshal.StringToPtr(deviceExtensions[i]);
        }

        PhysicalDeviceShaderAtomicFloatFeaturesEXT physicalDeviceShaderAtomicFloatFeaturesEXT = new()
        {
            SType = StructureType.PhysicalDeviceShaderAtomicFloatFeaturesExt,
            ShaderBufferFloat32AtomicAdd = Vk.True,
            ShaderBufferFloat32Atomics = Vk.True,
            ShaderSharedFloat32AtomicAdd = Vk.True,
            ShaderSharedFloat32Atomics = Vk.True,
        };

        PhysicalDeviceFeatures2 physicalDeviceFeatures2 = new()
        {
            SType = StructureType.PhysicalDeviceFeatures2,
            PNext = &physicalDeviceShaderAtomicFloatFeaturesEXT
        };

        DeviceCreateInfo deviceCreateInfo = new()
        {
            SType = StructureType.DeviceCreateInfo,
            QueueCreateInfoCount = 1,
            PQueueCreateInfos = &queueCreateInfo,
            EnabledExtensionCount = (uint)deviceExtensions.Length,
            PNext = &physicalDeviceFeatures2
        };
        
        // Set extension names
        fixed (IntPtr* extensionNamesPtr = extensionPtrs)
        {
            deviceCreateInfo.PpEnabledExtensionNames = (byte**)extensionNamesPtr;
        }

        Device device;
        var res = vk.CreateDevice(physicalDevice, &deviceCreateInfo, null, out device);
        if (res != Result.Success)
        {
            throw new Exception("Failed to create logical device.");
        }
        
        // Free allocated strings
        for (int i = 0; i < deviceExtensions.Length; i++)
        {
            Silk.NET.Core.Native.SilkMarshal.FreeString(extensionPtrs[i]);
        }

        return device;
    }
}