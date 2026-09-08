using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace CapKit.Editor
{
    internal static class CapKitRigObjects
    {
        internal static IEnumerable<CapKitFrameMarker> Owned(GameObject root) =>
            root.GetComponentsInChildren<CapKitFrameMarker>(true).Where(m => m.managedByCapKit);

        internal static Transform FindLink(GameObject root, string name)
        {
            var matches = root.GetComponentsInChildren<Transform>(true)
                .Where(t => t.name == name && !t.GetComponentsInParent<CapKitFrameMarker>(true).Any(m => m.managedByCapKit))
                .ToArray();
            if (matches.Length != 1)
                throw new InvalidOperationException("Expected one link '" + name + "', found " + matches.Length + ".");
            return matches[0];
        }

        internal static CapKitFrameMarker Mount(GameObject root, Transform parent, string name,
            Vector3 position, Quaternion rotation, HashSet<GameObject> keep)
        {
            var candidates = Owned(root).Where(m => m.name == name).ToArray();
            var marker = candidates.FirstOrDefault(m => m.transform.parent == parent) ?? candidates.FirstOrDefault();
            Transform direct = parent.Find(name);
            if (direct != null && (marker == null || direct != marker.transform))
            {
                // Adopt an empty mount by name, but never overwrite an unrelated sensor/object.
                if (direct.childCount != 0 || direct.GetComponents<Component>().Any(c => !(c is Transform) && !(c is CapKitFrameMarker)))
                    throw new InvalidOperationException("Unowned object conflicts with " + name + ".");
                marker = Component<CapKitFrameMarker>(direct.gameObject);
            }
            if (marker == null) marker = new GameObject(name).AddComponent<CapKitFrameMarker>();
            marker.managedByCapKit = true;
            marker.transform.SetParent(parent, false);
            marker.transform.localPosition = position;
            marker.transform.localRotation = rotation;
            marker.transform.localScale = Vector3.one;
            marker.gameObject.SetActive(true);
            keep.Add(marker.gameObject);
            return marker;
        }

        internal static int Prune(GameObject root, HashSet<GameObject> keep)
        {
            int removed = 0;
            // Children first, since deleting the hub may also delete its GNSS children.
            foreach (var marker in Owned(root).OrderByDescending(m => Depth(m.transform)).ToArray())
            {
                if (marker == null || keep.Contains(marker.gameObject)) continue;
                if (marker.gameObject == root) throw new InvalidOperationException("Refusing to delete the prefab root.");
                Object.DestroyImmediate(marker.gameObject);
                removed++;
            }
            return removed;
        }

        internal static T Component<T>(GameObject go) where T : Component
        {
            var components = go.GetComponents<T>();
            for (int i = components.Length - 1; i > 0; i--) Object.DestroyImmediate(components[i]);
            var result = components.Length == 0 ? go.AddComponent<T>() : components[0];
            if (result is Behaviour behaviour) behaviour.enabled = true;
            return result;
        }

        internal static void Remove<T>(GameObject go) where T : Component
        {
            foreach (var component in go.GetComponents<T>()) Object.DestroyImmediate(component);
        }

        internal static void RemoveDevice(CapKitFrameMarker marker)
        {
            if (marker.sensorInstance == null) return;
            if (marker.sensorInstance.transform.parent != marker.transform)
                throw new InvalidOperationException("Owned LiDAR instance moved outside " + marker.name + ".");
            Object.DestroyImmediate(marker.sensorInstance);
            marker.sensorInstance = null;
        }

        internal static void Edit(Object target, Action<SerializedObject> edit)
        {
            using (var so = new SerializedObject(target))
            {
                so.Update();
                edit(so);
                so.ApplyModifiedPropertiesWithoutUndo();
            }
            // Required for nested UnitySensors prefab overrides, including private inherited fields.
            if (PrefabUtility.IsPartOfPrefabInstance(target)) PrefabUtility.RecordPrefabInstancePropertyModifications(target);
            EditorUtility.SetDirty(target);
        }

        internal static SerializedProperty Property(SerializedObject so, string path)
        {
            var property = so.FindProperty(path);
            if (property == null) throw new InvalidOperationException(so.targetObject.GetType().Name + " is missing serialized field " + path);
            return property;
        }

        internal static void Frequency(Object sensor, float rate) => Edit(sensor, so => Property(so, "_frequency").floatValue = rate);

        internal static void Publisher(Object publisher, string topic, string frame, float rate)
        {
            Edit(publisher, so =>
            {
                Property(so, "_frequency").floatValue = rate;
                Property(so, "_topicName").stringValue = topic;
                Property(so, "_serializer._header._frame_id").stringValue = frame;
            });
        }

        private static int Depth(Transform transform)
        {
            int depth = 0;
            for (var parent = transform.parent; parent != null; parent = parent.parent) depth++;
            return depth;
        }
    }
}
