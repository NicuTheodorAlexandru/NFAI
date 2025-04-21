using System.ComponentModel.DataAnnotations;
using System.Reflection;

namespace NFAI.Vulkan.Shaders;

public static class EnumExtensions
{
    public static string GetDisplayName<T>(this T enumValue) where T : Enum
    {
        var memberInfo = enumValue.GetType().GetMember(enumValue.ToString());
        if (memberInfo.Length > 0)
        {
            var attribute = memberInfo[0].GetCustomAttribute<DisplayAttribute>();
            if (attribute != null)
            {
                return attribute.Name ?? enumValue.ToString();
            }
        }
        return enumValue.ToString();
    }

}