using UnityEngine;

/// <summary>
/// How fast the simulation runs where it runs headless (a Linux player on a lab host): every ``period`` seconds of
/// wall time one line "[CapPace] sim +X s in Y s wall (factor F), N frames (fps, the slowest frame), physics steps" -
/// a factor under 1 is a simulation slower than the machines it stands for (their controllers see it as time) - and
/// the garbage collector's collections in it, whether the slowest frame had one, and the managed heap: a frame over a
/// second stops every machine's kit at once (11-02, the chain's r32: a 1.18 s frame, five machines stopped for a stale
/// pose within 17 s; nothing else was logged in it).
/// CapCycleRecorder makes it on "pace_log_s": seconds.
/// </summary>
public class CapPaceLog : MonoBehaviour
{
    public float period = 10f;

    private float wall0, sim0, slowest;
    private int frames, steps, gc0, gcLast;
    private bool slowestCollected;

    private void Start()
    {
        wall0 = Time.realtimeSinceStartup;
        sim0 = Time.time;
        gc0 = gcLast = System.GC.CollectionCount(0);
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
                  $"{(slowestCollected ? "with" : "without")} one), managed heap {System.GC.GetTotalMemory(false) / 1048576.0:F0} MB");
        wall0 = Time.realtimeSinceStartup;
        sim0 = Time.time;
        frames = steps = 0;
        slowest = 0f;
        gc0 = gc;
    }
}
