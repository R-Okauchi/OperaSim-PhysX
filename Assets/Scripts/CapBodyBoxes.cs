using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// The simulator's machines carry a box on their root body (5 x 1 x 5 m round the hub, on the "Ignore Collision" layer:
/// it meets nothing, and the body's mass properties are made from it) that is no part of the machine a sensor sees. Every
/// raycast met it: the kit's LiDARs mapped its corners, 3.5-3.6 m from the swing axis and past the machine's self mask,
/// and zx200's guard stopped on its own box each time its swing turned its body (11-02, the chain's r29-r32; the dozer's
/// own box, 10-26, was left out of its prefab: CapD37pxiBuilder). Each such box is put on the "Ignore Raycast" layer, which
/// then meets nothing either: the machines move as they did (the box stays, and so do their mass properties), the LiDARs'
/// rays (Physics.DefaultRaycastLayers) pass through it to what is behind, and the soil's own casts, which take every layer
/// (CapSoilGeometry.CastDown), still meet it as they did.
/// </summary>
public static class CapBodyBoxes
{
    public const int Layer = 2;   // "Ignore Raycast": left out of Physics.DefaultRaycastLayers

    /// <summary>Every machine's body box in the scene (CapCycleRecorder, after the scenario's machines are made): the
    /// names of the machines whose boxes were hidden.</summary>
    public static List<string> HideAll()
    {
        Isolate();
        var hidden = new List<string>();
        foreach (var rb in Object.FindObjectsOfType<Rigidbody>())
        {
            if (Hide(rb)) hidden.Add(rb.name);
        }
        return hidden;
    }

    /// <summary>The layer meets nothing (nothing else of this project is on it: a box moved there collides as it did).</summary>
    public static void Isolate()
    {
        for (int l = 0; l < 32; l++) Physics.IgnoreLayerCollision(Layer, l, true);
    }

    /// <summary>A body's own box - a box collider on its Rigidbody's object, on a layer that meets nothing - moved to Layer
    /// with the object; an object with another collider of its own is left as it is (said). False when nothing moved.
    /// The layer must be isolated.</summary>
    public static bool Hide(Rigidbody rb)
    {
        var go = rb.gameObject;
        if (go.layer == Layer || !MeetsNothing(go.layer)) return false;
        var own = go.GetComponents<Collider>();
        bool box = false;
        foreach (var c in own) box |= c is BoxCollider;
        if (!box) return false;
        foreach (var c in own)
        {
            if (!(c is BoxCollider))
            {
                Debug.Log($"[{nameof(CapBodyBoxes)}] {go.name}: its body's box left as it is (a {c.GetType().Name} on it too)");
                return false;
            }
        }
        go.layer = Layer;
        return true;
    }

    /// <summary>Whether a layer meets none of the layers in use (the named ones).</summary>
    public static bool MeetsNothing(int layer)
    {
        for (int l = 0; l < 32; l++)
        {
            if (string.IsNullOrEmpty(LayerMask.LayerToName(l))) continue;
            if (!Physics.GetIgnoreLayerCollision(layer, l)) return false;
        }
        return true;
    }
}
