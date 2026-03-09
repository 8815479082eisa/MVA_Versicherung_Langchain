import React, { useCallback, useEffect, useMemo, useState } from "react";
import { RepoRelation, RepoRelationType, RepoTreeNode, repoTreeData } from "../data/repoTree";

const STORAGE_EXPANDED_KEY = "repoExplorer.expandedIds";
const STORAGE_SELECTED_KEY = "repoExplorer.selectedId";
const STORAGE_SEARCH_KEY = "repoExplorer.searchQuery";

interface TreeIndex {
  nodeById: Record<string, RepoTreeNode>;
  parentById: Record<string, string | null>;
  folderIds: string[];
}

function buildTreeIndex(root: RepoTreeNode): TreeIndex {
  const nodeById: Record<string, RepoTreeNode> = {};
  const parentById: Record<string, string | null> = {};
  const folderIds: string[] = [];

  const walk = (node: RepoTreeNode, parentId: string | null) => {
    nodeById[node.id] = node;
    parentById[node.id] = parentId;

    if (node.kind === "folder") {
      folderIds.push(node.id);
    }

    for (const child of node.children ?? []) {
      walk(child, node.id);
    }
  };

  walk(root, null);
  return { nodeById, parentById, folderIds };
}

function uniqueIds(ids: string[]): string[] {
  return Array.from(new Set(ids));
}

function readStorageArray(key: string, fallback: string[]): string[] {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) {
      return fallback;
    }
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return fallback;
    }
    return parsed.filter((item): item is string => typeof item === "string");
  } catch {
    return fallback;
  }
}

function readStorageString(key: string, fallback: string): string {
  try {
    const raw = localStorage.getItem(key);
    return raw && raw.trim() ? raw : fallback;
  } catch {
    return fallback;
  }
}

function collectAncestorFolderIds(nodeId: string, index: TreeIndex): string[] {
  const result: string[] = [];
  let current: string | null = nodeId;

  while (current) {
    const node = index.nodeById[current];
    if (node?.kind === "folder") {
      result.push(current);
    }
    current = index.parentById[current];
  }

  return uniqueIds(result);
}

function nodeMatchesQuery(node: RepoTreeNode, query: string): boolean {
  if (!query) {
    return true;
  }
  const haystack = [
    node.name,
    node.path,
    node.descriptionShort,
    ...(node.descriptionLong ?? []),
    ...(node.tags ?? []),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query);
}

function filterTree(node: RepoTreeNode, query: string): RepoTreeNode | null {
  if (!query) {
    return node;
  }

  const filteredChildren = (node.children ?? [])
    .map((child) => filterTree(child, query))
    .filter((child): child is RepoTreeNode => child !== null);

  if (nodeMatchesQuery(node, query) || filteredChildren.length > 0) {
    return {
      ...node,
      children: filteredChildren,
    };
  }

  return null;
}

interface TreeNodeProps {
  node: RepoTreeNode;
  depth: number;
  selectedId: string;
  expandedSet: Set<string>;
  forceExpand: boolean;
  onSelectNode: (nodeId: string, fromTreeClick: boolean) => void;
}

const TreeNode: React.FC<TreeNodeProps> = ({
  node,
  depth,
  selectedId,
  expandedSet,
  forceExpand,
  onSelectNode,
}) => {
  const hasChildren = (node.children?.length ?? 0) > 0;
  const isSelected = node.id === selectedId;
  const isExpanded = forceExpand || expandedSet.has(node.id);
  const indent = depth * 14;

  return (
    <div>
      <button
        type="button"
        onClick={() => onSelectNode(node.id, true)}
        className={`w-full text-left px-2 py-1 rounded-md transition-colors ${
          isSelected ? "bg-blue-50 border border-blue-200" : "hover:bg-slate-50 border border-transparent"
        }`}
        style={{ paddingLeft: `${indent + 8}px` }}
      >
        <span className="inline-flex items-center gap-2">
          <span className="w-4 text-xs text-slate-500">
            {hasChildren ? (isExpanded ? "-" : "+") : "."}
          </span>
          <span className="text-xs font-semibold text-slate-500">{node.kind === "folder" ? "DIR" : "FILE"}</span>
          <span className={`text-sm ${isSelected ? "text-blue-900 font-medium" : "text-slate-800"}`}>{node.name}</span>
        </span>
      </button>

      {hasChildren && isExpanded && (
        <div className="mt-0.5">
          {node.children?.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              selectedId={selectedId}
              expandedSet={expandedSet}
              forceExpand={forceExpand}
              onSelectNode={onSelectNode}
            />
          ))}
        </div>
      )}
    </div>
  );
};

function relationTypeLabel(type: RepoRelationType): string {
  if (type === "dependsOn") {
    return "Depends on";
  }
  if (type === "usedBy") {
    return "Used by";
  }
  return "Related";
}

const RepoExplorer: React.FC = () => {
  const index = useMemo(() => buildTreeIndex(repoTreeData), []);
  const defaultExpanded = useMemo(
    () => uniqueIds([repoTreeData.id, "src", "data", "frontend"]),
    []
  );

  const [expandedIds, setExpandedIds] = useState<string[]>(() =>
    readStorageArray(STORAGE_EXPANDED_KEY, defaultExpanded)
  );
  const [selectedId, setSelectedId] = useState<string>(() =>
    readStorageString(STORAGE_SELECTED_KEY, repoTreeData.id)
  );
  const [searchQuery, setSearchQuery] = useState<string>(() =>
    readStorageString(STORAGE_SEARCH_KEY, "")
  );
  const [copiedPath, setCopiedPath] = useState<string>("");

  useEffect(() => {
    const validExpanded = uniqueIds(
      expandedIds.filter((id) => index.nodeById[id]?.kind === "folder")
    );
    if (!validExpanded.includes(repoTreeData.id)) {
      validExpanded.unshift(repoTreeData.id);
    }
    if (validExpanded.length !== expandedIds.length) {
      setExpandedIds(validExpanded);
    }
  }, [expandedIds, index.nodeById]);

  useEffect(() => {
    if (!index.nodeById[selectedId]) {
      setSelectedId(repoTreeData.id);
    }
  }, [selectedId, index.nodeById]);

  useEffect(() => {
    localStorage.setItem(STORAGE_EXPANDED_KEY, JSON.stringify(expandedIds));
  }, [expandedIds]);

  useEffect(() => {
    localStorage.setItem(STORAGE_SELECTED_KEY, selectedId);
  }, [selectedId]);

  useEffect(() => {
    localStorage.setItem(STORAGE_SEARCH_KEY, searchQuery);
  }, [searchQuery]);

  const selectedNode = index.nodeById[selectedId] ?? repoTreeData;
  const forceExpand = searchQuery.trim().length > 0;
  const filteredRoot = useMemo(
    () => filterTree(repoTreeData, searchQuery.trim().toLowerCase()),
    [searchQuery]
  );
  const expandedSet = useMemo(() => new Set(expandedIds), [expandedIds]);

  const breadcrumb = useMemo(() => {
    const items: RepoTreeNode[] = [];
    let currentId: string | null = selectedNode.id;

    while (currentId) {
      const node = index.nodeById[currentId];
      if (!node) {
        break;
      }
      items.push(node);
      currentId = index.parentById[currentId];
    }
    return items.reverse();
  }, [selectedNode.id, index.nodeById, index.parentById]);

  const selectNode = useCallback(
    (nodeId: string, fromTreeClick: boolean) => {
      const node = index.nodeById[nodeId];
      if (!node) {
        return;
      }

      setSelectedId(nodeId);

      const ancestorFolders = collectAncestorFolderIds(nodeId, index);
      setExpandedIds((prev) => uniqueIds([...prev, ...ancestorFolders]));

      if (fromTreeClick && node.kind === "folder") {
        setExpandedIds((prev) =>
          prev.includes(nodeId) ? prev.filter((id) => id !== nodeId) : uniqueIds([...prev, nodeId])
        );
      }
    },
    [index]
  );

  const handleExpandAll = useCallback(() => {
    setExpandedIds(uniqueIds(index.folderIds));
  }, [index.folderIds]);

  const handleCollapseAll = useCallback(() => {
    setExpandedIds([repoTreeData.id]);
  }, []);

  const handleCopyPath = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(selectedNode.path);
      setCopiedPath(selectedNode.path);
      setTimeout(() => setCopiedPath(""), 1800);
    } catch {
      setCopiedPath("");
    }
  }, [selectedNode.path]);

  const relationList: RepoRelation[] = useMemo(() => {
    const fromIds = (selectedNode.relatedIds ?? []).map((id) => ({
      id,
      type: "related" as const,
    }));
    const merged = [...(selectedNode.relations ?? []), ...fromIds];
    return merged.filter((relation) => Boolean(index.nodeById[relation.id]));
  }, [selectedNode, index.nodeById]);

  const relationGroups = useMemo(
    () => ({
      dependsOn: relationList.filter((item) => item.type === "dependsOn"),
      usedBy: relationList.filter((item) => item.type === "usedBy"),
      related: relationList.filter((item) => item.type === "related"),
    }),
    [relationList]
  );

  const RelationBlock: React.FC<{ title: string; items: RepoRelation[] }> = ({ title, items }) => {
    if (items.length === 0) {
      return null;
    }

    return (
      <div>
        <h4 className="text-sm font-semibold text-slate-800">{title}</h4>
        <div className="mt-2 space-y-2">
          {items.map((relation) => {
            const target = index.nodeById[relation.id];
            if (!target) {
              return null;
            }

            return (
              <button
                key={`${title}-${relation.id}`}
                type="button"
                onClick={() => selectNode(relation.id, false)}
                className="w-full text-left rounded-md border border-slate-200 px-3 py-2 hover:bg-slate-50"
              >
                <div className="text-sm font-medium text-blue-700">{target.path}</div>
                <div className="text-xs text-slate-600">
                  {relation.note ? relation.note : relationTypeLabel(relation.type)}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
      <div className="grid grid-cols-1 lg:grid-cols-12 min-h-[700px]">
        <aside className="lg:col-span-5 border-r border-slate-200 p-4">
          <div className="flex items-center justify-between gap-2 mb-3">
            <h2 className="text-lg font-semibold text-slate-900">Repository Map</h2>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleExpandAll}
                className="text-xs px-2 py-1 rounded border border-slate-300 hover:bg-slate-50"
              >
                Expand all
              </button>
              <button
                type="button"
                onClick={handleCollapseAll}
                className="text-xs px-2 py-1 rounded border border-slate-300 hover:bg-slate-50"
              >
                Collapse all
              </button>
            </div>
          </div>

          <input
            type="text"
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search path, tags, purpose..."
            className="w-full mb-3 rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />

          <div className="h-[600px] overflow-auto pr-1">
            {filteredRoot ? (
              <TreeNode
                node={filteredRoot}
                depth={0}
                selectedId={selectedNode.id}
                expandedSet={expandedSet}
                forceExpand={forceExpand}
                onSelectNode={selectNode}
              />
            ) : (
              <div className="text-sm text-slate-600">No matching nodes for this search.</div>
            )}
          </div>
        </aside>

        <section className="lg:col-span-7 p-6 bg-slate-50/40">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-xl font-semibold text-slate-900 break-all">{selectedNode.path}</h3>
              <p className="text-sm text-slate-600 mt-1">
                Type: <span className="font-medium">{selectedNode.kind}</span>
              </p>
            </div>
            <button
              type="button"
              onClick={handleCopyPath}
              className="text-sm px-3 py-1.5 rounded-md border border-slate-300 bg-white hover:bg-slate-50"
            >
              {copiedPath === selectedNode.path ? "Path copied" : "Copy path"}
            </button>
          </div>

          <div className="mt-3 text-sm text-slate-600">
            {breadcrumb.map((node, indexInList) => (
              <span key={`crumb-${node.id}`}>
                {indexInList > 0 ? " / " : ""}
                <button
                  type="button"
                  onClick={() => selectNode(node.id, false)}
                  className="hover:text-blue-700"
                >
                  {node.name}
                </button>
              </span>
            ))}
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            {selectedNode.tags.map((tag) => (
              <span
                key={`${selectedNode.id}-${tag}`}
                className="text-xs px-2 py-1 rounded-full bg-blue-100 text-blue-800 border border-blue-200"
              >
                {tag}
              </span>
            ))}
          </div>

          <div className="mt-6 space-y-6">
            <div>
              <h4 className="text-sm font-semibold text-slate-800">Purpose / Responsibility</h4>
              <p className="mt-2 text-sm text-slate-700">{selectedNode.descriptionShort}</p>
              <ul className="mt-3 list-disc list-inside space-y-1 text-sm text-slate-700">
                {selectedNode.descriptionLong.map((line) => (
                  <li key={`${selectedNode.id}-${line}`}>{line}</li>
                ))}
              </ul>
            </div>

            {selectedNode.entryPoints && selectedNode.entryPoints.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-slate-800">Key entry points</h4>
                <ul className="mt-2 list-disc list-inside space-y-1 text-sm text-slate-700">
                  {selectedNode.entryPoints.map((entryPoint) => (
                    <li key={`${selectedNode.id}-ep-${entryPoint}`}>
                      <code className="bg-white border border-slate-200 rounded px-1 py-0.5">{entryPoint}</code>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {selectedNode.runtimeArtifacts && selectedNode.runtimeArtifacts.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-slate-800">Related runtime artifacts</h4>
                <ul className="mt-2 list-disc list-inside space-y-1 text-sm text-slate-700">
                  {selectedNode.runtimeArtifacts.map((artifactPath) => (
                    <li key={`${selectedNode.id}-artifact-${artifactPath}`}>
                      <code className="bg-white border border-slate-200 rounded px-1 py-0.5">{artifactPath}</code>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-slate-800">Used by / depends on</h4>
              <RelationBlock title="Depends on" items={relationGroups.dependsOn} />
              <RelationBlock title="Used by" items={relationGroups.usedBy} />
              <RelationBlock title="Related" items={relationGroups.related} />
              {relationList.length === 0 && <p className="text-sm text-slate-600">No links for this node yet.</p>}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default RepoExplorer;
