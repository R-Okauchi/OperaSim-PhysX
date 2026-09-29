using System;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEngine;

namespace CapKit.Editor
{
    /// <summary>
    /// Keeps the editor running at full rate while it plays the simulation with another app in front.
    ///
    /// UnitySensors samples and publishes once per frame (UnitySensor.Update / RosMsgPublisher.Update). With the editor
    /// behind another window (VS Code in front, a screen-sharing session) macOS App-Napped it and the frames fell to
    /// 6-30 fps: the kit IMU dropped from 100 Hz to 6 Hz, the kit estimate degraded past its gate and the machines
    /// stopped, as the safety layer must (live 2026-09-29 run57, run59). While playing, this holds a latency-critical,
    /// user-initiated NSProcessInfo activity (no App Nap) and turns vSync off (an occluded window's vblank no longer
    /// paces the loop; the frame rate is capped at 120). Both end when play mode ends.
    /// </summary>
    [InitializeOnLoad]
    static class CapKeepAwake
    {
        const string ObjC = "/usr/lib/libobjc.A.dylib";
        // NSActivityOptions (Foundation/NSProcessInfo.h)
        const ulong IdleSystemSleepDisabled = 1UL << 20;
        const ulong UserInitiated = 0x00FFFFFFUL | IdleSystemSleepDisabled;
        const ulong LatencyCritical = 0xFF00000000UL;

        static IntPtr _activity = IntPtr.Zero;
        static int _vSyncBefore = -1;
        static int _targetBefore = int.MinValue;

        [DllImport(ObjC)] static extern IntPtr objc_getClass(string name);
        [DllImport(ObjC)] static extern IntPtr sel_registerName(string name);
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] static extern IntPtr Send(IntPtr receiver, IntPtr selector);
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] static extern IntPtr SendString(IntPtr receiver, IntPtr selector, string utf8);
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] static extern IntPtr SendActivity(IntPtr receiver, IntPtr selector, ulong options, IntPtr reason);
        [DllImport(ObjC, EntryPoint = "objc_msgSend")] static extern void SendObject(IntPtr receiver, IntPtr selector, IntPtr argument);

        static CapKeepAwake()
        {
            EditorApplication.playModeStateChanged += OnPlayModeChanged;
            if (EditorApplication.isPlaying)
                Begin();                       // a domain reload while playing
        }

        static void OnPlayModeChanged(PlayModeStateChange change)
        {
            if (change == PlayModeStateChange.EnteredPlayMode)
                Begin();
            else if (change == PlayModeStateChange.ExitingPlayMode)
                End();
        }

        static void Begin()
        {
            if (_vSyncBefore < 0)
            {
                _vSyncBefore = QualitySettings.vSyncCount;
                _targetBefore = Application.targetFrameRate;
            }
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 120;
            if (Application.platform != RuntimePlatform.OSXEditor || _activity != IntPtr.Zero)
                return;
            try
            {
                IntPtr info = Send(objc_getClass("NSProcessInfo"), sel_registerName("processInfo"));
                IntPtr reason = SendString(objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"),
                                           "CAP simulation playing (sensors sample per frame)");
                IntPtr activity = SendActivity(info, sel_registerName("beginActivityWithOptions:reason:"),
                                               UserInitiated | LatencyCritical, reason);
                if (activity != IntPtr.Zero)
                {
                    Send(activity, sel_registerName("retain"));
                    _activity = activity;
                    Debug.Log("[CapKeepAwake] App Nap held off and vSync off while playing");
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("[CapKeepAwake] could not hold off App Nap: " + e.Message);
            }
        }

        static void End()
        {
            if (_vSyncBefore >= 0)
            {
                QualitySettings.vSyncCount = _vSyncBefore;
                Application.targetFrameRate = _targetBefore;
                _vSyncBefore = -1;
            }
            if (_activity == IntPtr.Zero)
                return;
            try
            {
                IntPtr info = Send(objc_getClass("NSProcessInfo"), sel_registerName("processInfo"));
                SendObject(info, sel_registerName("endActivity:"), _activity);
                Send(_activity, sel_registerName("release"));
            }
            catch (Exception e)
            {
                Debug.LogWarning("[CapKeepAwake] could not end the activity: " + e.Message);
            }
            _activity = IntPtr.Zero;
        }
    }
}
