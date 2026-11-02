using System.Collections.Generic;
using Stopwatch = System.Diagnostics.Stopwatch;
using System.Text;
using UnityEngine;
using UnityEngine.LowLevel;

/// <summary>
/// How fast the simulation runs where it runs headless (a Linux player on a lab host): every ``period`` seconds of
/// wall time one line "[CapPace] sim +X s in Y s wall (factor F), N frames (fps, the slowest frame), physics steps" -
/// a factor under 1 is a simulation slower than the machines it stands for (their controllers see it as time) - and
/// the garbage collector's collections in it, whether the slowest frame had one, and the managed heap: a frame over a
/// second stops every machine's kit at once (11-02, the chain's r32: a 1.18 s frame, five machines stopped for a stale
/// pose within 17 s; nothing else was logged in it). The slowest frame's time is said by the player loop's phases
/// (Initialization, EarlyUpdate, FixedUpdate, PreUpdate, Update, PreLateUpdate, PostLateUpdate and the wait before the
/// next frame), a phase a collection fell in marked: r36-r37's frames of 0.9-2.7 s came with no slow soil sync, with a
/// collection and without one.
/// CapCycleRecorder makes it on "pace_log_s": seconds.
/// </summary>
public class CapPaceLog : MonoBehaviour
{
    public float period = 10f;

    private float wall0, sim0, slowest;
    private int frames, steps, gc0, gcLast;
    private bool slowestCollected;

    // the player loop's phases: the tick each began at this frame (and the last one's end), the collections by then
    private static readonly Stopwatch Clock = Stopwatch.StartNew();
    private static string[] phases;
    private static long[] marks;
    private static int[] collections;
    private static long frameTicks;              // the slowest frame of the period, by its phases
    private static long[] frameParts;
    private static bool[] frameCollected;

    private void Start()
    {
        wall0 = Time.realtimeSinceStartup;
        sim0 = Time.time;
        gc0 = gcLast = System.GC.CollectionCount(0);
        Instrument();
    }

    /// <summary>A mark at the start of each top-level phase of the player loop and one at the end of the last: once
    /// a player.</summary>
    private static void Instrument()
    {
        if (phases != null)
            return;
        var loop = PlayerLoop.GetCurrentPlayerLoop();
        var top = loop.subSystemList;
        if (top == null || top.Length == 0)
            return;
        phases = new string[top.Length];
        marks = new long[top.Length + 1];
        collections = new int[top.Length + 1];
        frameParts = new long[top.Length + 1];
        frameCollected = new bool[top.Length + 1];
        for (int i = 0; i < top.Length; i++)
        {
            int k = i;
            phases[i] = top[i].type != null ? top[i].type.Name : $"phase {i}";
            var list = new List<PlayerLoopSystem>(top[i].subSystemList ?? new PlayerLoopSystem[0]);
            list.Insert(0, new PlayerLoopSystem { type = typeof(CapPaceLog), updateDelegate = () => Mark(k) });
            if (i == top.Length - 1)
                list.Add(new PlayerLoopSystem { type = typeof(CapPaceLog), updateDelegate = () => Mark(top.Length) });
            top[i].subSystemList = list.ToArray();
        }
        loop.subSystemList = top;
        PlayerLoop.SetPlayerLoop(loop);
    }

    private static void Mark(int k)
    {
        long now = Clock.ElapsedTicks;
        int gc = System.GC.CollectionCount(0);
        if (k == 0 && marks[marks.Length - 1] > marks[0])
        {
            // the frame before this one, whole: its phases and the wait from its end to now
            long total = now - marks[0];
            if (total > frameTicks)
            {
                frameTicks = total;
                for (int i = 0; i < phases.Length; i++)
                {
                    frameParts[i] = marks[i + 1] - marks[i];
                    frameCollected[i] = collections[i + 1] != collections[i];
                }
                frameParts[phases.Length] = now - marks[phases.Length];
                frameCollected[phases.Length] = gc != collections[phases.Length];
            }
        }
        marks[k] = now;
        collections[k] = gc;
    }

    private static string SlowestFrame()
    {
        if (phases == null || frameTicks == 0)
            return "";
        double ms = 1000.0 / Stopwatch.Frequency;
        var s = new StringBuilder($"; the slowest frame {frameTicks * ms:F0} ms:");
        for (int i = 0; i <= phases.Length; i++)
        {
            if (frameParts[i] * ms < 1.0 && !frameCollected[i])
                continue;
            s.Append($" {(i < phases.Length ? phases[i] : "the wait")} {frameParts[i] * ms:F0}");
            if (frameCollected[i])
                s.Append(" (a collection)");
        }
        frameTicks = 0;
        return s.ToString();
    }

    private void FixedUpdate() => steps++;

    private void Update()
    {
        frames++;
        // a collection since the last frame was in the time this frame's delta measures
        int gc = System.GC.CollectionCount(0);
        if (Time.unscaledDeltaTime > slowest)
        {
            slowest = Time.unscaledDeltaTime;
            slowestCollected = gc != gcLast;
        }
        gcLast = gc;
        float wall = Time.realtimeSinceStartup - wall0;
        if (wall < period)
            return;
        float sim = Time.time - sim0;
        Debug.Log($"[CapPace] sim +{sim:F1} s in {wall:F1} s wall (factor {sim / wall:F2}), {frames} frames " +
                  $"({frames / wall:F1} fps, the slowest {slowest * 1000f:F0} ms), {steps} physics steps " +
                  $"({steps / Mathf.Max(sim, 1e-3f):F1} per sim s), {gc - gc0} collections (the slowest frame " +
                  $"{(slowestCollected ? "with" : "without")} one), managed heap {System.GC.GetTotalMemory(false) / 1048576.0:F0} MB" +
                  SlowestFrame());
        wall0 = Time.realtimeSinceStartup;
        sim0 = Time.time;
        frames = steps = 0;
        slowest = 0f;
        gc0 = gc;
    }
}
