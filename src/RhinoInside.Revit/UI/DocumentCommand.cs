using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;

namespace RhinoInside.Revit.UI
{
  static class Extension
  {
    public static bool ActivateRibbonTab(this UIApplication application, string tabName)
    {
      var ribbon = Autodesk.Windows.ComponentManager.Ribbon;
      foreach (var tab in ribbon.Tabs)
      {
        if (tab.Name == tabName)
        {
          tab.IsActive = true;
          return true;
        }
      }

      return false;
    }

    internal static PushButton AddPushButton(this RibbonPanel ribbonPanel, Type commandType, string text = null, string tooltip = null, Type availability = null)
    {
      var buttonData = new PushButtonData
      (
        commandType.Name,
        text ?? commandType.Name,
        commandType.Assembly.Location,
        commandType.FullName
      );

      if (ribbonPanel.AddItem(buttonData) is PushButton pushButton)
      {
        pushButton.ToolTip = tooltip;
        if (availability != null)
          pushButton.AvailabilityClassName = availability.FullName;

        return pushButton;
      }

      return null;
    }

    internal static PushButton AddPushButton(this PulldownButton pullDownButton, Type commandType, string text = null, string tooltip = null, Type availability = null)
    {
      var buttonData = new PushButtonData
      (
        commandType.Name,
        text ?? commandType.Name,
        commandType.Assembly.Location,
        commandType.FullName
      );

      if (pullDownButton.AddPushButton(buttonData) is PushButton pushButton)
      {
        pushButton.ToolTip = tooltip;
        if (availability != null)
          pushButton.AvailabilityClassName = availability.FullName;

        return pushButton;
      }

      return null;
    }
  }

  abstract public class Command : External.UI.Command
  {
    public static PushButtonData NewPushButtonData<CommandType>(string text = null)
    where CommandType : IExternalCommand
    {
      return new PushButtonData
      (
        typeof(CommandType).Name,
        text ?? typeof(CommandType).Name,
        typeof(CommandType).Assembly.Location,
        typeof(CommandType).FullName
      );
    }

    public static PushButtonData NewPushButtonData<CommandType, AvailabilityType>(string text = null)
    where CommandType : IExternalCommand where AvailabilityType : IExternalCommandAvailability
    {
      return new PushButtonData
      (
        typeof(CommandType).Name,
        text ?? typeof(CommandType).Name,
        typeof(CommandType).Assembly.Location,
        typeof(CommandType).FullName
      )
      {
        AvailabilityClassName = typeof(AvailabilityType).FullName
      };
    }

    public static ToggleButtonData NewToggleButtonData<CommandType, AvailabilityType>(string text = null)
    where CommandType : IExternalCommand where AvailabilityType : IExternalCommandAvailability
    {
      return new ToggleButtonData
      (
        typeof(CommandType).Name,
        text ?? typeof(CommandType).Name,
        typeof(CommandType).Assembly.Location,
        typeof(CommandType).FullName
      )
      {
        AvailabilityClassName = typeof(AvailabilityType).FullName
      };
    }

    public class AllwaysAvailable : IExternalCommandAvailability
    {
      bool IExternalCommandAvailability.IsCommandAvailable(UIApplication app, CategorySet selectedCategories) => true;
    }

    protected override bool CatchException(Exception e, UIApplication app)
    {
      if (app.LoadedApplications.OfType<Addin>().FirstOrDefault() is Addin addin)
        return addin.CatchException(e, app, this);

      return base.CatchException(e, app);
    }
  }

  abstract public class DocumentCommand : Command
  {
    public class Availability : External.UI.CommandAvailability
    {
      // We can not relay on the UIApplication first argument.
      // Seams other Add-ins are calling this method with wrong values.
      // I add the try-catch just because this is called many times.
      public override bool IsCommandAvailable(UIApplication _, CategorySet selectedCategories)
      {
        try  { return Revit.ActiveUIDocument?.Document?.IsValidObject ?? false; }
        catch (Autodesk.Revit.Exceptions.ApplicationException) { return false; }
      }
    }
  }
}