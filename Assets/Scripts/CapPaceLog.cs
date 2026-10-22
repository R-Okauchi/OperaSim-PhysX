using UnityEngine;

/// <summary>
/// How fast the simulation runs where it runs headless (a Linux player on a lab host): every ``period`` seconds of
/// wall time one line "[CapPace] sim +X s in Y s wall (factor F), N frames (fps, the slowest frame), physics steps" -
/// a factor under 1 is a simulation slower than the machines it stands for (their controllers see it as time).
/// CapCycleRecorder makes it on "pace_log_s": seconds.
/// </summary>
public class CapPaceLog : MonoBehaviour
{
    public float period = 10f;

    private float wall0, sim0, slowest;
    private int frames, steps;

    private void Start()
    {
        wall0 = Time.realtimeSinceStartup;
        sim0 = Time.time;
    }

    private void FixedUpdate() => steps++;

    private void Update()
    {
        frames++;
        slowest = Mathf.Max(slowest, Time.unscaledDeltaTime);
        float wall = Time.realtimeSinceStartup - wall0;
        if (wall < period)
            return;
        float sim = Time.time - sim0;
        Debug.Log($"[CapPace] sim +{sim:F1} s in {wall:F1} s wall (factor {sim / wall:F2}), {frames} frames " +
                  $"({frames / wall:F1} fps, the slowest {slowest * 1000f:F0} ms), {steps} physics steps " +
                  $"({steps / Mathf.Max(sim, 1e-3f):F1} per sim s)");
        wall0 = Time.realtimeSinceStartup;
        sim0 = Time.time;
        frames = steps = 0;
        slowest = 0f;
    }
}
