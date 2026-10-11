/**
 * features/preparation/PrepareAreaPage.tsx
 *
 * Prepare Analysis Area staging workflow:
 *   1. Define an inclusion / AOI polygon (draw on map)
 *   2. Add / edit / delete manual exclusion polygons
 *   3. Edit exclusion category + comment
 *   4. Preview effective area metrics
 *   5. Save Draft → Confirm & Lock → Unlock
 *
 * Mode-aware behaviour
 * --------------------
 * VEGETATION MODE  – single scene; no T1/T2 selector, no T2 data, no wetland
 *                    boundary requirement.  The single scene is loaded as "t1"
 *                    internally (implementation detail) but the UI presents it
 *                    as "Vegetation Scene".
 *
 * WETLAND MODE     – two-date workflow; T1 / T2 tab selector is shown;
 *                    wetland boundary is optional.
 *
 * Coordinate system
 * -----------------
 * Datasets that lack a geographic CRS are presented in pixel space
 * (origin = bottom-left of raster, Y-up, matching OpenLayers convention).
 * The backend exclusion.py normalises OL [x, y] → raster row/col on read.
 */

import { useEffect, useRef, useState, useCallback } from "react";
import Map from "ol/Map";
import View from "ol/View";
import VectorLayer from "ol/layer/Vector";
import VectorSource from "ol/source/Vector";
import ImageLayer from "ol/layer/Image";
import ImageStatic from "ol/source/ImageStatic";
import Draw from "ol/interaction/Draw";
import Modify from "ol/interaction/Modify";
import GeoJSON from "ol/format/GeoJSON";
import Projection from "ol/proj/Projection";
import { addProjection, get as getProjection } from "ol/proj";
import { Fill, Stroke, Style, Circle as CircleStyle } from "ol/style";
import type Feature from "ol/Feature";
import type Geometry from "ol/geom/Geometry";
import "ol/ol.css";

import { useProject } from "../../context/useProject";
import type { ManualExclusion } from "../../context/ProjectContext";
import type { PreparationMetrics } from "../../api/client";
import { getDatasetPreviewImageUrl, getDatasetPreviewInfo } from "../../api/client";

// ─── types ───────────────────────────────────────────────────────────────────

type AnalysisMode = "vegetation" | "wetland";

export interface PrepareAreaPageProps {
  /** Shared analysis mode from AppShell – controls which UI branch is rendered. */
  analysisMode: AnalysisMode;
}

// ─── pixel projection ────────────────────────────────────────────────────────

const DEFAULT_W = 2592;
const DEFAULT_H = 1944;

function getPixelProjection(width: number, height: number) {
  const code = `GIS-PREP-PX-${width}x${height}`;
  let proj = getProjection(code);
  if (!proj) {
    proj = new Projection({
      code,
      units: "pixels",
      extent: [0, 0, width, height],
    });
    addProjection(proj);
  }
  return proj;
}

// ─── OL styles ───────────────────────────────────────────────────────────────

const inclStyle = new Style({
  fill: new Fill({ color: "rgba(54,92,69,0.15)" }),
  stroke: new Stroke({ color: "#365c45", width: 2.5, lineDash: [6, 4] }),
});

const exclStyle = new Style({
  fill: new Fill({ color: "rgba(155,59,50,0.18)" }),
  stroke: new Stroke({ color: "#9b3b32", width: 2 }),
});

const exclSelectedStyle = new Style({
  fill: new Fill({ color: "rgba(155,59,50,0.35)" }),
  stroke: new Stroke({ color: "#9b3b32", width: 3 }),
});

const drawStyle = new Style({
  fill: new Fill({ color: "rgba(100,160,120,0.2)" }),
  stroke: new Stroke({ color: "#4caf80", width: 2, lineDash: [4, 4] }),
  image: new CircleStyle({
    radius: 5,
    fill: new Fill({ color: "#4caf80" }),
  }),
});

// ─── helpers ─────────────────────────────────────────────────────────────────

const GJ = new GeoJSON();

function toGeom(feat: Feature<Geometry>) {
  return JSON.parse(GJ.writeGeometry(feat.getGeometry()!));
}

function fromGeom(geom: Record<string, unknown>) {
  return GJ.readFeature({ type: "Feature", geometry: geom }) as Feature<Geometry>;
}

function fmtHa(v?: number | null) {
  return v == null ? "—" : `${v.toFixed(2)} ha`;
}
function fmtPx(v?: number | null) {
  return v == null ? "—" : Number(v).toLocaleString();
}

// ─── ExclusionRow ─────────────────────────────────────────────────────────────

const CATEGORIES = [
  "Infrastructure", "Water Body", "Urban Area",
  "Agricultural", "Road / Track", "Other",
];

type ExclusionRowProps = {
  excl: ManualExclusion;
  selected: boolean;
  locked: boolean;
  onSelect: () => void;
  onUpdate: (u: ManualExclusion) => void;
  onDelete: () => void;
};

function ExclusionRow({ excl, selected, locked, onSelect, onUpdate, onDelete }: ExclusionRowProps) {
  const [editing, setEditing] = useState(false);
  const [cat, setCat] = useState(excl.category);
  const [cmt, setCmt] = useState(excl.comment);

  const save = () => {
    onUpdate({ ...excl, category: cat.trim(), comment: cmt.trim() });
    setEditing(false);
  };

  return (
    <div
      className={[
        "rounded-md border p-3 cursor-pointer transition-colors",
        selected
          ? "border-[var(--danger)] bg-red-500/5"
          : "border-[var(--border)] bg-[var(--background)] hover:bg-[var(--surface)]",
      ].join(" ")}
      onClick={onSelect}
    >
      {editing && !locked ? (
        <div className="space-y-2" onClick={(e) => e.stopPropagation()}>
          <select
            value={cat}
            onChange={(e) => setCat(e.target.value)}
            className="w-full rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-1 text-xs text-[var(--text)] outline-none focus:border-[var(--accent)]"
          >
            {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
          </select>
          <textarea
            rows={2}
            value={cmt}
            onChange={(e) => setCmt(e.target.value)}
            placeholder="Comment (optional)"
            className="w-full resize-none rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-1 text-xs text-[var(--text)] outline-none focus:border-[var(--accent)]"
          />
          <div className="flex justify-end gap-2">
            <button type="button" onClick={() => setEditing(false)}
              className="px-3 py-1 text-xs rounded border border-[var(--border)] text-[var(--text-secondary)]">
              Cancel
            </button>
            <button type="button" onClick={save}
              className="px-3 py-1 text-xs rounded bg-[var(--accent)] text-white hover:opacity-90">
              Save
            </button>
          </div>
        </div>
      ) : (
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="text-xs font-semibold text-[var(--danger)]">
              {excl.category || "Uncategorised"}
            </div>
            {excl.comment && (
              <div className="mt-0.5 text-[11px] text-[var(--text-secondary)] leading-snug">{excl.comment}</div>
            )}
          </div>
          {!locked && (
            <div className="flex shrink-0 gap-1">
              <button type="button" title="Edit"
                onClick={(e) => { e.stopPropagation(); setEditing(true); }}
                className="rounded px-2 py-0.5 text-[10px] border border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--surface)]">
                ✎
              </button>
              <button type="button" title="Delete"
                onClick={(e) => { e.stopPropagation(); onDelete(); }}
                className="rounded px-2 py-0.5 text-[10px] border border-red-500/30 text-[var(--danger)] hover:bg-red-500/10">
                ✕
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── MetricTile ───────────────────────────────────────────────────────────────

function MetricTile({ label, ha, px, accent }: { label: string; ha?: number | null; px?: number | null; accent?: boolean }) {
  return (
    <div className={[
      "rounded-md border p-2.5",
      accent ? "border-[var(--accent)]/40 bg-[var(--accent-soft)]" : "border-[var(--border)] bg-[var(--background)]",
    ].join(" ")}>
      <div className="text-[10px] uppercase tracking-widest text-[var(--text-secondary)]">{label}</div>
      <div className={["mt-0.5 text-sm font-semibold", accent ? "text-[var(--accent)]" : "text-[var(--text)]"].join(" ")}>
        {fmtHa(ha)}
      </div>
      <div className="text-[10px] text-[var(--text-secondary)]">{fmtPx(px)} px</div>
    </div>
  );
}

// ─── PrepareAreaPage ──────────────────────────────────────────────────────────

type DrawMode = "none" | "inclusion" | "exclusion";

export default function PrepareAreaPage({ analysisMode }: PrepareAreaPageProps) {
  const {
    project,
    preparation,
    manualExclusions,
    fetchPreparation,
    saveDraft,
    commitArea,
    unlockArea,
    addManualExclusion,
    updateManualExclusion,
    removeManualExclusion,
  } = useProject();

  const isVegetationMode = analysisMode === "vegetation";

  // raster dimensions
  const [rasterW, setRasterW] = useState(DEFAULT_W);
  const [rasterH, setRasterH] = useState(DEFAULT_H);

  /**
   * In VEGETATION MODE the preview is always "t1" (the single vegetation scene).
   * In WETLAND MODE the user can toggle between t1 and t2.
   * We keep `previewTp` as the internal source identifier; it is only exposed
   * as a UI toggle in wetland mode.
   */
  const [previewTp, setPreviewTp] = useState<"t1" | "t2">("t1");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // local UI
  const [drawMode, setDrawMode] = useState<DrawMode>("none");
  const [selectedExclId, setSelectedExclId] = useState<string | null>(null);
  const [inclusionGeom, setInclusionGeom] = useState<Record<string, unknown> | null>(null);
  const [pendingCat, setPendingCat] = useState(CATEGORIES[0]);
  const [pendingCmt, setPendingCmt] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isCommitting, setIsCommitting] = useState(false);
  const [isUnlocking, setIsUnlocking] = useState(false);
  const [opError, setOpError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<PreparationMetrics | null>(null);

  // OL refs
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<Map | null>(null);
  const imageLayerRef = useRef<ImageLayer<ImageStatic> | null>(null);
  const viewFittedRef = useRef(false);
  const inclSrcRef = useRef<VectorSource | null>(null);
  const exclSrcRef = useRef<VectorSource | null>(null);
  const drawRef = useRef<Draw | null>(null);
  const modifyRef = useRef<Modify | null>(null);

  const isLocked = preparation?.is_locked ?? false;
  const isCommitted = preparation?.status === "COMMITTED";
  const hasDraft = preparation?.status === "DRAFT" || inclusionGeom !== null;

  // When switching to vegetation mode, always force previewTp back to "t1"
  // so stale "t2" selection from a prior wetland session does not persist.
  useEffect(() => {
    if (isVegetationMode) {
      setPreviewTp("t1");
    }
  }, [isVegetationMode]);

  // ── fetch on mount ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!project) return;
    viewFittedRef.current = false;

    fetchPreparation(project.id).then((prep) => {
      if (prep?.inclusion) setInclusionGeom(prep.inclusion as Record<string, unknown>);
      if (prep?.metrics) setMetrics(prep.metrics);
    });

    getDatasetPreviewInfo(project.id)
      .then((info) => {
        if (info.width && info.height) {
          setRasterW(info.width);
          setRasterH(info.height);
        }
      })
      .catch(() => {/* use defaults */});

    setPreviewUrl(getDatasetPreviewImageUrl(project.id, "t1"));
  }, [project?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!project) return;
    // In vegetation mode, always use "t1" regardless of previewTp state.
    const tp = isVegetationMode ? "t1" : previewTp;
    setPreviewUrl(getDatasetPreviewImageUrl(project.id, tp));
  }, [previewTp, project?.id, isVegetationMode]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── sync metrics after preparation update ─────────────────────────────────
  useEffect(() => {
    if (preparation?.metrics) setMetrics(preparation.metrics);
  }, [preparation?.metrics]);

  // ── init OL map ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const inclSrc = new VectorSource();
    const exclSrc = new VectorSource();
    inclSrcRef.current = inclSrc;
    exclSrcRef.current = exclSrc;

    const imgLayer = new ImageLayer<ImageStatic>({
      zIndex: 1,
      opacity: 0.9,
    });
    imageLayerRef.current = imgLayer;

    const proj = getPixelProjection(rasterW, rasterH);

    const map = new Map({
      target: mapRef.current,
      layers: [
        imgLayer,
        new VectorLayer({ source: inclSrc, style: () => inclStyle, zIndex: 2 }),
        new VectorLayer({
          source: exclSrc,
          style: (feat) =>
            feat.get("exclusionId") === selectedExclId
              ? exclSelectedStyle
              : exclStyle,
          zIndex: 3,
        }),
      ],
      view: new View({
        projection: proj,
        center: [rasterW / 2, rasterH / 2],
        zoom: 1,
        minZoom: -3,
        maxZoom: 10,
      }),
    });

    mapInstanceRef.current = map;

    return () => {
      map.dispose();
      mapInstanceRef.current = null;
      imageLayerRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── sync raster preview layer to map ──────────────────────────────────────
  useEffect(() => {
    const map = mapInstanceRef.current;
    const imgLayer = imageLayerRef.current;
    if (!map || !imgLayer || !previewUrl || rasterW <= 0 || rasterH <= 0) return;

    const proj = getPixelProjection(rasterW, rasterH);
    const source = new ImageStatic({
      url: previewUrl,
      imageExtent: [0, 0, rasterW, rasterH],
      projection: proj,
    });

    imgLayer.setSource(source);

    const currentProj = map.getView().getProjection();
    const projCode = proj.getCode();
    const currentCode = currentProj?.getCode();

    if (!viewFittedRef.current || currentCode !== projCode) {
      const view = new View({
        projection: proj,
        center: [rasterW / 2, rasterH / 2],
        minZoom: -3,
        maxZoom: 10,
      });
      map.setView(view);
      view.fit([0, 0, rasterW, rasterH], { padding: [30, 30, 30, 30] });
      viewFittedRef.current = true;
    }
  }, [previewUrl, rasterW, rasterH]);

  // ── sync inclusion to map ─────────────────────────────────────────────────
  useEffect(() => {
    const src = inclSrcRef.current;
    if (!src) return;
    src.clear();
    if (inclusionGeom) {
      try { src.addFeature(fromGeom(inclusionGeom)); } catch { /* ignore */ }
    }
  }, [inclusionGeom]);

  // ── sync exclusions to map ────────────────────────────────────────────────
  useEffect(() => {
    const src = exclSrcRef.current;
    if (!src) return;
    src.clear();
    for (const ex of manualExclusions) {
      try {
        const f = fromGeom(ex.geometry as Record<string, unknown>);
        f.set("exclusionId", ex.id);
        src.addFeature(f);
      } catch { /* skip */ }
    }
  }, [manualExclusions]);

  // ── interaction helpers ───────────────────────────────────────────────────
  const clearInteractions = useCallback(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    if (drawRef.current) { map.removeInteraction(drawRef.current); drawRef.current = null; }
    if (modifyRef.current) { map.removeInteraction(modifyRef.current); modifyRef.current = null; }
  }, []);

  const startDraw = useCallback((mode: "inclusion" | "exclusion") => {
    const map = mapInstanceRef.current;
    if (!map) return;
    clearInteractions();

    const src = mode === "inclusion" ? inclSrcRef.current! : exclSrcRef.current!;
    const draw = new Draw({ source: src, type: "Polygon", style: drawStyle });

    draw.on("drawend", (e) => {
      const geom = toGeom(e.feature as Feature<Geometry>);
      if (mode === "inclusion") {
        inclSrcRef.current!.clear();
        inclSrcRef.current!.addFeature(e.feature as Feature<Geometry>);
        setInclusionGeom(geom);
      } else {
        const id = `excl-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        (e.feature as Feature<Geometry>).set("exclusionId", id);
        addManualExclusion({ id, geometry: geom, category: pendingCat, comment: pendingCmt });
      }
      map.removeInteraction(draw);
      drawRef.current = null;
      setDrawMode("none");
    });

    map.addInteraction(draw);
    drawRef.current = draw;
    setDrawMode(mode);
  }, [addManualExclusion, clearInteractions, pendingCat, pendingCmt]);

  const startModify = useCallback(() => {
    const map = mapInstanceRef.current;
    const src = inclSrcRef.current;
    if (!map || !src) return;
    clearInteractions();
    const modify = new Modify({ source: src });
    modify.on("modifyend", () => {
      const f = src.getFeatures()[0];
      if (f) setInclusionGeom(toGeom(f as Feature<Geometry>));
    });
    map.addInteraction(modify);
    modifyRef.current = modify;
  }, [clearInteractions]);

  const handleResetInclusion = useCallback(() => {
    inclSrcRef.current?.clear();
    setInclusionGeom(null);
    clearInteractions();
    setDrawMode("none");
  }, [clearInteractions]);

  const handleDeleteExcl = useCallback((id: string) => {
    const feat = exclSrcRef.current?.getFeatures().find((f) => f.get("exclusionId") === id);
    if (feat) exclSrcRef.current?.removeFeature(feat);
    removeManualExclusion(id);
    if (selectedExclId === id) setSelectedExclId(null);
  }, [removeManualExclusion, selectedExclId]);

  // ── workflow actions ──────────────────────────────────────────────────────
  const handleSaveDraft = useCallback(async () => {
    setIsSaving(true);
    setOpError(null);
    try {
      await saveDraft(inclusionGeom as Record<string, unknown> | null, manualExclusions);
    } catch (err: unknown) {
      setOpError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setIsSaving(false);
    }
  }, [inclusionGeom, manualExclusions, saveDraft]);

  const handleCommit = useCallback(async () => {
    setIsCommitting(true);
    setOpError(null);
    try {
      if (inclusionGeom || manualExclusions.length > 0) {
        await saveDraft(inclusionGeom as Record<string, unknown> | null, manualExclusions);
      }
      await commitArea();
    } catch (err: unknown) {
      setOpError(err instanceof Error ? err.message : "Commit failed");
    } finally {
      setIsCommitting(false);
    }
  }, [commitArea, inclusionGeom, manualExclusions, saveDraft]);

  const handleUnlock = useCallback(async () => {
    setIsUnlocking(true);
    setOpError(null);
    try {
      await unlockArea();
    } catch (err: unknown) {
      setOpError(err instanceof Error ? err.message : "Unlock failed");
    } finally {
      setIsUnlocking(false);
    }
  }, [unlockArea]);

  // ── status badge ──────────────────────────────────────────────────────────
  const statusLabel = preparation?.status ?? "NOT_PREPARED";
  const statusBadge =
    statusLabel === "COMMITTED"
      ? "text-[var(--success)] border-[var(--success)]/40 bg-[var(--success)]/10"
      : statusLabel === "DRAFT"
      ? "text-[var(--warning)] border-[var(--warning)]/40 bg-[var(--warning)]/10"
      : "text-[var(--text-secondary)] border-[var(--border)] bg-[var(--background)]";

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-5 h-full min-h-0">

      {/* Header */}
      <section>
        <div className="flex items-start justify-between">
          <div>
            <div className="text-2xl font-semibold tracking-tight text-[var(--text)]">
              Prepare Analysis Area
            </div>
            <div className="mt-1 text-sm text-[var(--text-secondary)]">
              {isVegetationMode
                ? "Define the inclusion polygon and mark exclusions for single-date vegetation health analysis."
                : "Define the inclusion polygon and mark exclusions before locking the dataset for analysis."}
            </div>
          </div>
          <span className={["rounded-md border px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em]", statusBadge].join(" ")}>
            {statusLabel.replace("_", " ")}
          </span>
        </div>
      </section>

      {opError && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">{opError}</div>
      )}

      {isLocked && (
        <div className="flex items-center justify-between rounded-md border border-[var(--warning)]/40 bg-[var(--warning)]/10 p-3">
          <span className="text-xs text-[var(--warning)] font-medium">Preparation is locked. Unlock to make edits.</span>
          <button
            type="button"
            disabled={isUnlocking}
            onClick={handleUnlock}
            className="ml-4 rounded-md border border-[var(--warning)]/40 px-4 py-1.5 text-xs font-medium text-[var(--warning)] hover:bg-[var(--warning)]/10 disabled:opacity-50"
          >
            {isUnlocking ? "Unlocking…" : "Unlock"}
          </button>
        </div>
      )}

      {/* Two-column layout */}
      <div className="grid grid-cols-[1fr_340px] gap-5 flex-1 min-h-0">

        {/* ── Map panel ───────────────────────────────────────────────────── */}
        <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] flex flex-col overflow-hidden">

          {/* Toolbar */}
          <div className="flex flex-wrap items-center gap-2 border-b border-[var(--border)] px-4 py-2.5">

            {/* ── Scene / temporal selector ─────────────────────────────── */}
            {isVegetationMode ? (
              /* VEGETATION MODE: single scene label — no T1/T2 toggle */
              <div
                className="rounded px-2 py-0.5 text-[10px] font-semibold uppercase border border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                data-testid="vegetation-scene-label"
              >
                Vegetation Scene
              </div>
            ) : (
              /* WETLAND MODE: T1 / T2 toggle */
              <div className="flex items-center gap-1" data-testid="wetland-t1-t2-selector">
                <span className="text-[10px] uppercase tracking-widest text-[var(--text-secondary)] mr-1">Base</span>
                {(["t1", "t2"] as const).map((tp) => (
                  <button key={tp} type="button" onClick={() => setPreviewTp(tp)}
                    className={[
                      "rounded px-2 py-0.5 text-[10px] font-semibold uppercase border transition-colors",
                      previewTp === tp
                        ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                        : "border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--background)]",
                    ].join(" ")}>
                    {tp.toUpperCase()}
                  </button>
                ))}
              </div>
            )}

            <div className="h-4 w-px bg-[var(--border)]" />

            {/* AOI tools */}
            <button type="button" disabled={isLocked}
              onClick={() => drawMode === "inclusion" ? (clearInteractions(), setDrawMode("none")) : startDraw("inclusion")}
              className={[
                "rounded-md border px-3 py-1.5 text-xs font-medium transition-colors disabled:opacity-40",
                drawMode === "inclusion"
                  ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
                  : "border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--background)]",
              ].join(" ")}>
              {drawMode === "inclusion" ? "● Drawing AOI…" : inclusionGeom ? "✎ Redraw AOI" : "+ Draw AOI"}
            </button>

            {inclusionGeom && (
              <button type="button" disabled={isLocked} onClick={startModify}
                className="rounded-md border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] hover:bg-[var(--background)] disabled:opacity-40">
                ⊹ Edit AOI
              </button>
            )}

            {inclusionGeom && (
              <button type="button" disabled={isLocked} onClick={handleResetInclusion}
                className="rounded-md border border-red-500/30 px-3 py-1.5 text-xs font-medium text-[var(--danger)] hover:bg-red-500/5 disabled:opacity-40">
                ✕ Clear AOI
              </button>
            )}

            <div className="h-4 w-px bg-[var(--border)]" />

            {/* Exclusion draw */}
            <button type="button" disabled={isLocked}
              onClick={() => drawMode === "exclusion" ? (clearInteractions(), setDrawMode("none")) : startDraw("exclusion")}
              className={[
                "rounded-md border px-3 py-1.5 text-xs font-medium transition-colors disabled:opacity-40",
                drawMode === "exclusion"
                  ? "border-[var(--danger)] bg-red-500/10 text-[var(--danger)]"
                  : "border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--background)]",
              ].join(" ")}>
              {drawMode === "exclusion" ? "● Drawing excl…" : "+ Add Exclusion"}
            </button>
          </div>

          {/* Map canvas */}
          <div className="relative flex-1 min-h-0 bg-[#121614] overflow-hidden">
            <div ref={mapRef} className="absolute inset-0" style={{ zIndex: 1 }} />
            {drawMode !== "none" && (
              <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-[var(--surface)]/90 border border-[var(--border)] px-4 py-1.5 text-xs font-medium text-[var(--text)] shadow-lg z-10 pointer-events-none">
                {drawMode === "inclusion"
                  ? "Click to place vertices — double-click to close the polygon"
                  : "Click to place exclusion vertices — double-click to finish"}
              </div>
            )}
          </div>

          {/* Legend */}
          <div className="flex items-center gap-4 border-t border-[var(--border)] px-4 py-2 text-[10px] text-[var(--text-secondary)]">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-4 rounded-sm border border-dashed border-[var(--accent)] bg-[var(--accent)]/15" />
              Inclusion / AOI
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-4 rounded-sm border border-[var(--danger)] bg-[var(--danger)]/15" />
              Exclusion
            </span>
            <span className="ml-auto">Scroll to zoom · Drag to pan</span>
          </div>
        </div>

        {/* ── Right panel ──────────────────────────────────────────────────── */}
        <div className="flex flex-col gap-4 min-h-0 overflow-y-auto">

          {/* Next exclusion settings */}
          {!isLocked && (
            <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <div className="mb-3 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                Next Exclusion Settings
              </div>
              <label className="block mb-2">
                <span className="text-xs font-medium text-[var(--text)]">Category</span>
                <select
                  value={pendingCat}
                  onChange={(e) => setPendingCat(e.target.value)}
                  className="mt-1 w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1.5 text-xs text-[var(--text)] outline-none focus:border-[var(--accent)]"
                >
                  {CATEGORIES.map((c) => <option key={c}>{c}</option>)}
                </select>
              </label>
              <label className="block">
                <span className="text-xs font-medium text-[var(--text)]">Comment</span>
                <textarea
                  rows={2}
                  value={pendingCmt}
                  onChange={(e) => setPendingCmt(e.target.value)}
                  placeholder="Optional description…"
                  className="mt-1 w-full resize-none rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1.5 text-xs text-[var(--text)] outline-none focus:border-[var(--accent)]"
                />
              </label>
            </div>
          )}

          {/* Exclusions list */}
          <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 flex flex-col" style={{ minHeight: "160px" }}>
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                Exclusions ({manualExclusions.length})
              </div>
              {manualExclusions.length > 0 && !isLocked && (
                <button type="button"
                  onClick={() => {
                    exclSrcRef.current?.clear();
                    [...manualExclusions].forEach((ex) => removeManualExclusion(ex.id));
                    setSelectedExclId(null);
                  }}
                  className="text-[10px] text-[var(--danger)] hover:underline">
                  Clear all
                </button>
              )}
            </div>
            <div className="flex flex-col gap-2 overflow-y-auto flex-1">
              {manualExclusions.length === 0 ? (
                <div className="rounded-md border border-dashed border-[var(--border)] p-4 text-center text-xs text-[var(--text-secondary)]">
                  {isLocked
                    ? "No exclusions defined."
                    : `No exclusions yet. Click "+ Add Exclusion" above.`}
                </div>
              ) : (
                manualExclusions.map((ex) => (
                  <ExclusionRow
                    key={ex.id}
                    excl={ex}
                    locked={isLocked}
                    selected={selectedExclId === ex.id}
                    onSelect={() => setSelectedExclId((p) => p === ex.id ? null : ex.id)}
                    onUpdate={updateManualExclusion}
                    onDelete={() => handleDeleteExcl(ex.id)}
                  />
                ))
              )}
            </div>
          </div>

          {/* Metrics */}
          {metrics && (
            <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <div className="mb-3 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                Area Preview
              </div>
              <div className="grid grid-cols-2 gap-2">
                <MetricTile label="Source" ha={metrics.source_ha} px={metrics.source_pixels} />
                <MetricTile label="Inclusion" ha={metrics.inclusion_ha} px={metrics.inclusion_pixels} />
                <MetricTile label="Excluded" ha={metrics.excluded_ha} px={metrics.excluded_pixels} />
                <MetricTile label="Effective" ha={metrics.effective_ha} px={metrics.effective_pixels} accent />
              </div>
              <div className="mt-2 text-[10px] text-[var(--text-secondary)]">
                Basis: {(metrics.area_basis ?? "—").replace(/_/g, " ")}
              </div>
            </div>
          )}

          {/* Action buttons */}
          <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 space-y-3">
            <div className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
              Workflow Actions
            </div>

            <button type="button" disabled={isLocked || isSaving} onClick={handleSaveDraft}
              className="w-full rounded-md border border-[var(--border)] px-4 py-2.5 text-xs font-medium text-[var(--text)] hover:bg-[var(--background)] disabled:opacity-40 transition-colors">
              {isSaving ? "Saving…" : "💾  Save Draft"}
            </button>

            {!isCommitted && (
              <button type="button" disabled={isLocked || isCommitting || !hasDraft} onClick={handleCommit}
                title={!hasDraft ? "Save a draft first" : undefined}
                className="w-full rounded-md bg-[var(--accent)] px-4 py-2.5 text-xs font-medium text-white hover:opacity-90 disabled:opacity-40 transition-opacity">
                {isCommitting ? "Locking…" : "🔒  Confirm & Lock"}
              </button>
            )}

            {isCommitted && (
              <>
                <button type="button" disabled={isUnlocking} onClick={handleUnlock}
                  className="w-full rounded-md border border-[var(--warning)]/50 px-4 py-2.5 text-xs font-medium text-[var(--warning)] hover:bg-[var(--warning)]/10 disabled:opacity-40 transition-colors">
                  {isUnlocking ? "Unlocking…" : "🔓  Unlock & Edit"}
                </button>
                <div className="rounded-md border border-[var(--success)]/40 bg-[var(--success)]/10 p-2.5 text-xs text-[var(--success)]">
                  ✓ Preparation locked — ready to run analysis.
                </div>
              </>
            )}

            {!isCommitted && !hasDraft && (
              <p className="text-center text-[10px] text-[var(--text-secondary)]">
                Draw an inclusion polygon and save a draft before committing.
              </p>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
