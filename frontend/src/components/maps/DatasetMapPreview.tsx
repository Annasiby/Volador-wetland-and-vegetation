import { useEffect, useRef, useState } from "react";
import Map from "ol/Map";
import View from "ol/View";
import WebGLTileLayer from "ol/layer/WebGLTile";
import VectorLayer from "ol/layer/Vector";
import VectorSource from "ol/source/Vector";
import GeoTIFF from "ol/source/GeoTIFF";
import Draw from "ol/interaction/Draw";
import Modify from "ol/interaction/Modify";
import GeoJSON from "ol/format/GeoJSON";
import Projection from "ol/proj/Projection";
import { addProjection } from "ol/proj";
import { Fill, Stroke, Style } from "ol/style";
import { useProject } from "../../context/useProject";
import "ol/ol.css";
import type { ManualExclusion } from "../../context/ProjectContext";

type Props = {
  t1File: File | null;
  t2File: File | null;
};

/*
 * The uploaded DJI TIFFs currently have:
 *
 *   width  = 2592
 *   height = 1944
 *   CRS    = None
 *   transform = identity
 *
 * Therefore this preview intentionally uses image/pixel coordinates.
 *
 * This is NOT a geographic CRS and must not be interpreted as EPSG:4326.
 *
 * IMPORTANT — axis orientation:
 *   OpenLayers map coordinates for this projection are Y-up (origin at
 *   the bottom-left of PIXEL_EXTENT), because every OL projection is
 *   Cartesian. Raster/pixel space (numpy rows, rasterio row/col) is
 *   Y-down (origin top-left, row 0 = top of the image).
 *
 *   That means the [x, y] coordinates written out by GeoJSON here are
 *   NOT the same as raster row/col — y is inverted relative to row.
 *   The backend (wetland/exclusion.py, build_manual_exclusion_mask)
 *   is responsible for flipping y -> (height - y) before rasterizing
 *   against the identity-transform grid. Do not "fix" this on the
 *   frontend by flipping coordinates here as well, or the two flips
 *   will cancel out and reintroduce the bug.
 */
const RASTER_WIDTH = 2592;
const RASTER_HEIGHT = 1944;

const PIXEL_EXTENT = [
  0,
  0,
  RASTER_WIDTH,
  RASTER_HEIGHT,
] as [number, number, number, number];

const PIXEL_PROJECTION = new Projection({
  code: "GIS-PIXEL",
  units: "pixels",
  extent: PIXEL_EXTENT,
});

addProjection(PIXEL_PROJECTION);

export default function DatasetMapPreview({
  t1File,
  t2File,
}: Props) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<Map | null>(null);

  const rasterLayerRef =
    useRef<WebGLTileLayer | null>(null);

  const exclusionSourceRef =
    useRef<VectorSource | null>(null);

  const drawRef =
    useRef<Draw | null>(null);

  const modifyRef =
    useRef<Modify | null>(null);

  const objectUrlRef =
    useRef<string | null>(null);

  const {
    manualExclusions,
    addManualExclusion,
    updateManualExclusion,
    removeManualExclusion,
  } = useProject();

  const [selectedDate, setSelectedDate] =
    useState<"t1" | "t2">("t1");

  const [drawing, setDrawing] =
    useState(false);

  const [status, setStatus] = useState(
    "Upload a T1 or T2 raster to preview it."
  );

  /*
   * Exclusion currently being drawn and waiting
   * for category/comment information.
   */
  const [pendingExclusion, setPendingExclusion] =
    useState<ManualExclusion | null>(null);

  const [exclusionCategory, setExclusionCategory] =
    useState("");

  const [exclusionComment, setExclusionComment] =
    useState("");

  const [formError, setFormError] =
    useState("");

  const selectedFile =
    selectedDate === "t1"
      ? t1File
      : t2File;

  /*
   * ------------------------------------------------------------
   * Initialize map
   * ------------------------------------------------------------
   */

  useEffect(() => {
    if (!mapRef.current) {
      return;
    }

    const exclusionSource =
      new VectorSource();

    const exclusionLayer =
      new VectorLayer({
        source: exclusionSource,
        zIndex: 20,
        style: new Style({
          fill: new Fill({
            color: "rgba(220, 38, 38, 0.28)",
          }),
          stroke: new Stroke({
            color: "#dc2626",
            width: 3,
          }),
        }),
      });

    const map = new Map({
      target: mapRef.current,

      layers: [
        exclusionLayer,
      ],

      view: new View({
        projection: PIXEL_PROJECTION,
        center: [
          RASTER_WIDTH / 2,
          RASTER_HEIGHT / 2,
        ],
        zoom: 1,
        minZoom: -5,
        maxZoom: 8,
        extent: PIXEL_EXTENT,
        showFullExtent: true,
      }),
    });

    mapInstanceRef.current = map;
    exclusionSourceRef.current =
      exclusionSource;

    return () => {
      if (drawRef.current) {
        map.removeInteraction(
          drawRef.current
        );
        drawRef.current = null;
      }

      if (modifyRef.current) {
        map.removeInteraction(
          modifyRef.current
        );
        modifyRef.current = null;
      }

      if (objectUrlRef.current) {
        URL.revokeObjectURL(
          objectUrlRef.current
        );
        objectUrlRef.current = null;
      }

      map.setTarget(undefined);
      mapInstanceRef.current = null;
    };
  }, []);

  /*
   * ------------------------------------------------------------
   * Load selected TIFF
   * ------------------------------------------------------------
   */

  useEffect(() => {
    const map =
      mapInstanceRef.current;

    if (!map) {
      return;
    }

    if (rasterLayerRef.current) {
      map.removeLayer(
        rasterLayerRef.current
      );

      rasterLayerRef.current = null;
    }

    if (objectUrlRef.current) {
      URL.revokeObjectURL(
        objectUrlRef.current
      );

      objectUrlRef.current = null;
    }

    if (!selectedFile) {
      setStatus(
        "Upload a T1 or T2 raster to preview it."
      );
      return;
    }

    const objectUrl =
      URL.createObjectURL(selectedFile);

    objectUrlRef.current =
      objectUrl;

    setStatus(
      `Loading ${selectedDate.toUpperCase()} raster...`
    );

    const source = new GeoTIFF({
      sources: [
        {
          url: objectUrl,
        },
      ],

      projection:
        PIXEL_PROJECTION,

      normalize: true,

      interpolate: true,
    });

    const rasterLayer =
      new WebGLTileLayer({
        source,
        opacity: 1,
        zIndex: 1,
        extent: PIXEL_EXTENT,
      });

    map.addLayer(rasterLayer);

    rasterLayerRef.current =
      rasterLayer;

    const currentMap =
      mapInstanceRef.current;

    if (currentMap) {
      currentMap.setView(
        new View({
          projection:
            PIXEL_PROJECTION,

          center: [
            RASTER_WIDTH / 2,
            RASTER_HEIGHT / 2,
          ],

          zoom: 1,

          minZoom: -5,

          maxZoom: 8,

          extent:
            PIXEL_EXTENT,

          showFullExtent: true,
        })
      );

      currentMap
        .getView()
        .fit(
          PIXEL_EXTENT,
          {
            padding: [
              30,
              30,
              30,
              30,
            ],
            duration: 300,
            maxZoom: 2,
          }
        );
    }

    const handleSourceError =
      () => {
        setStatus(
          `Unable to render ${selectedDate.toUpperCase()} raster.`
        );
      };

    source.on(
      "error",
      handleSourceError
    );

    const readyTimer =
      window.setTimeout(() => {
        if (
          mapInstanceRef.current &&
          rasterLayerRef.current ===
            rasterLayer
        ) {
          setStatus(
            `${selectedDate.toUpperCase()} raster loaded`
          );

          mapInstanceRef.current
            .renderSync();
        }
      }, 500);

    return () => {
      window.clearTimeout(
        readyTimer
      );

      source.un(
        "error",
        handleSourceError
      );

      if (
        rasterLayerRef.current ===
        rasterLayer
      ) {
        map.removeLayer(
          rasterLayer
        );

        rasterLayerRef.current =
          null;
      }

      if (
        objectUrlRef.current ===
        objectUrl
      ) {
        URL.revokeObjectURL(
          objectUrl
        );

        objectUrlRef.current =
          null;
      }
    };
  }, [
    selectedFile,
    selectedDate,
  ]);

  /*
   * ------------------------------------------------------------
   * Restore saved manual exclusions
   * ------------------------------------------------------------
   */

  useEffect(() => {
    const source =
      exclusionSourceRef.current;

    if (!source) {
      return;
    }

    source.clear();

    if (
      manualExclusions.length ===
      0
    ) {
      return;
    }

    const format =
      new GeoJSON();

    for (
      const exclusion of
      manualExclusions
    ) {
      try {
        const feature =
          format.readFeature(
            {
              type: "Feature",

              properties: {
                id: exclusion.id,
                category:
                  exclusion.category,
                comment:
                  exclusion.comment,
              },

              geometry:
                exclusion.geometry,
            }
          );

        if (Array.isArray(feature)) {
          source.addFeatures(
            feature
          );
        } else {
          source.addFeature(
            feature
          );
        }
      } catch (error) {
        console.error(
          "Failed to render exclusion:",
          exclusion.id,
          error
        );
      }
    }
  }, [
    manualExclusions,
  ]);

  /*
   * ------------------------------------------------------------
   * Start drawing
   * ------------------------------------------------------------
   */

  const startDrawing =
    () => {
      const map =
        mapInstanceRef.current;

      const source =
        exclusionSourceRef.current;

      if (
        !map ||
        !source ||
        drawing
      ) {
        return;
      }

      if (!selectedFile) {
        setStatus(
          "Upload a dataset before drawing an exclusion."
        );
        return;
      }

      if (drawRef.current) {
        map.removeInteraction(
          drawRef.current
        );

        drawRef.current =
          null;
      }

      if (modifyRef.current) {
        map.removeInteraction(
          modifyRef.current
        );

        modifyRef.current =
          null;
      }

      const draw =
        new Draw({
          source,
          type: "Polygon",
        });

      draw.on(
        "drawend",
        (event) => {
          const geometry =
            event.feature.getGeometry();

          if (!geometry) {
            setDrawing(false);
            return;
          }

          const format =
            new GeoJSON();

          const feature =
            format.writeFeatureObject(
              event.feature
            );

          if (
            !feature.geometry ||
            (
              feature.geometry.type !==
                "Polygon" &&
              feature.geometry.type !==
                "MultiPolygon"
            )
          ) {
            setDrawing(false);
            return;
          }

          /*
          * The map uses GIS-PIXEL coordinates.
          *
          * These are OpenLayers map coordinates (Y-up, origin at the
          * bottom-left of PIXEL_EXTENT) for the unreferenced DJI
          * raster — they are NOT yet raster row/col (Y-down, origin
          * top-left). Do not flip X, translate, or offset the
          * geometry here; the backend applies the Y-axis flip when
          * rasterizing against the analysis grid (see
          * wetland/exclusion.py::build_manual_exclusion_mask).
          */
          console.log(
            "MANUAL EXCLUSION DRAW COORDINATES:",
            feature.geometry
          );

          const exclusion: ManualExclusion = {
            id: `exc_${Date.now()}`,

            geometry:
              feature.geometry as
                | {
                    type: "Polygon";
                    coordinates:
                      number[][][];
                  }
                | {
                    type: "MultiPolygon";
                    coordinates:
                      number[][][][];
                  },

            category: "",

            comment: "",
          };

          /*
          * Keep the polygon visible while the user
          * enters its category/comment.
          */
          setPendingExclusion(
            exclusion
          );

          setExclusionCategory(
            ""
          );

          setExclusionComment(
            ""
          );

          setFormError("");

          setDrawing(false);

          map.removeInteraction(
            draw
          );

          drawRef.current =
            null;

          setStatus(
            "Add a category and comment for this exclusion."
          );
        }
      );

      map.addInteraction(
        draw
      );

      drawRef.current =
        draw;

      setDrawing(true);

      setStatus(
        "Click points around the area to exclude. Double-click to finish."
      );
    };

  /*
   * ------------------------------------------------------------
   * Save pending exclusion
   * ------------------------------------------------------------
   */

  const savePendingExclusion =
    () => {
      if (!pendingExclusion) {
        return;
      }

      const category =
        exclusionCategory.trim();

      const comment =
        exclusionComment.trim();

      if (!category) {
        setFormError(
          "Select a category."
        );
        return;
      }

      if (!comment) {
        setFormError(
          "A comment is required."
        );
        return;
      }

      addManualExclusion({
        ...pendingExclusion,
        category,
        comment,
      });

      setPendingExclusion(
        null
      );

      setExclusionCategory(
        ""
      );

      setExclusionComment(
        ""
      );

      setFormError("");

      setStatus(
        "Exclusion area saved."
      );
    };

  /*
   * ------------------------------------------------------------
   * Cancel pending exclusion
   * ------------------------------------------------------------
   */

  const cancelPendingExclusion =
    () => {
      const source =
        exclusionSourceRef.current;

      if (
        source &&
        pendingExclusion
      ) {
        const feature =
          source
            .getFeatures()
            .find(
              (item) =>
                item.get("id") ===
                pendingExclusion.id
            );

        if (feature) {
          source.removeFeature(
            feature
          );
        }
      }

      setPendingExclusion(
        null
      );

      setExclusionCategory(
        ""
      );

      setExclusionComment(
        ""
      );

      setFormError("");

      setStatus(
        "Exclusion discarded."
      );
    };

  /*
   * ------------------------------------------------------------
   * Cancel drawing
   * ------------------------------------------------------------
   */

  const cancelDrawing =
    () => {
      const map =
        mapInstanceRef.current;

      if (!map) {
        return;
      }

      if (drawRef.current) {
        map.removeInteraction(
          drawRef.current
        );

        drawRef.current =
          null;
      }

      setDrawing(false);

      setStatus(
        "Drawing cancelled."
      );
    };

  /*
   * ------------------------------------------------------------
   * Modify existing exclusions
   * ------------------------------------------------------------
   */

  const enableModify =
    () => {
      const map =
        mapInstanceRef.current;

      const source =
        exclusionSourceRef.current;

      if (!map || !source) {
        return;
      }

      if (
        manualExclusions.length ===
        0
      ) {
        return;
      }

      if (modifyRef.current) {
        map.removeInteraction(
          modifyRef.current
        );
      }

      const modify =
        new Modify({
          source,
        });

      modify.on(
        "modifyend",
        (event) => {
          const format =
            new GeoJSON();

          event.features.forEach(
            (feature) => {
              const id =
                feature.get("id");

              if (!id) {
                return;
              }

              const featureObject =
                format.writeFeatureObject(
                  feature
                );

              const geometry =
                featureObject.geometry;

              if (
                !geometry ||
                (
                  geometry.type !==
                    "Polygon" &&
                  geometry.type !==
                    "MultiPolygon"
                )
              ) {
                return;
              }

              const existing =
                manualExclusions.find(
                  (exclusion) =>
                    exclusion.id ===
                    id
                );

              if (!existing) {
                return;
              }

              updateManualExclusion({
                ...existing,

                geometry:
                  geometry as
                    | {
                        type: "Polygon";
                        coordinates:
                          number[][][];
                      }
                    | {
                        type: "MultiPolygon";
                        coordinates:
                          number[][][][];
                      },
              });
            }
          );

          setStatus(
            "Exclusion area updated."
          );
        }
      );

      map.addInteraction(
        modify
      );

      modifyRef.current =
        modify;

      setStatus(
        "Drag exclusion vertices to edit the area."
      );
    };

  /*
   * ------------------------------------------------------------
   * Render
   * ------------------------------------------------------------
   */

  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-sm font-semibold text-[var(--text)]">
            Dataset Spatial Preview
          </div>

          <div className="mt-1 text-xs text-[var(--text-secondary)]">
            Inspect the uploaded raster and define areas
            that should be excluded from analysis.
          </div>
        </div>

        <div className="flex rounded-md border border-[var(--border)] bg-[var(--background)] p-1">
          <button
            type="button"
            onClick={() =>
              setSelectedDate("t1")
            }
            className={`rounded px-3 py-1.5 text-xs font-medium ${
              selectedDate === "t1"
                ? "bg-[var(--accent)] text-white"
                : "text-[var(--text-secondary)]"
            }`}
          >
            T1
          </button>

          <button
            type="button"
            onClick={() =>
              setSelectedDate("t2")
            }
            className={`rounded px-3 py-1.5 text-xs font-medium ${
              selectedDate === "t2"
                ? "bg-[var(--accent)] text-white"
                : "text-[var(--text-secondary)]"
            }`}
          >
            T2
          </button>
        </div>
      </div>

      <div className="relative mt-4 h-[520px] overflow-hidden rounded-lg border border-[var(--border)] bg-[#ECECE8]">
        <div
          ref={mapRef}
          className="absolute inset-0 h-full w-full"
        />

        <div className="absolute left-4 top-4 z-30 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 shadow-md">
          <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            Manual Exclusions
          </div>

          <div className="mt-2 flex gap-2">
            {!drawing ? (
              <button
                type="button"
                onClick={
                  startDrawing
                }
                disabled={
                  !selectedFile ||
                  pendingExclusion !==
                    null
                }
                className="rounded-md bg-[var(--accent)] px-3 py-2 text-xs font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
              >
                Exclude Area
              </button>
            ) : (
              <button
                type="button"
                onClick={
                  cancelDrawing
                }
                className="rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-xs font-medium text-[var(--text)]"
              >
                Cancel
              </button>
            )}

            <button
              type="button"
              onClick={
                enableModify
              }
              disabled={
                manualExclusions.length ===
                  0 ||
                pendingExclusion !==
                  null
              }
              className="rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-xs font-medium text-[var(--text)] disabled:cursor-not-allowed disabled:opacity-40"
            >
              Edit Areas
            </button>
          </div>

          <div className="mt-2 text-[11px] text-[var(--text-secondary)]">
            {manualExclusions.length ===
            0
              ? "No exclusions defined"
              : `${manualExclusions.length} exclusion${
                  manualExclusions.length ===
                  1
                    ? ""
                    : "s"
                } defined`}
          </div>

          {manualExclusions.length >
            0 && (
            <div className="mt-3 max-h-[180px] space-y-2 overflow-y-auto">
              {manualExclusions.map(
                (
                  exclusion,
                  index
                ) => (
                  <div
                    key={
                      exclusion.id
                    }
                    className="rounded-md border border-[var(--border)] bg-[var(--background)] p-2.5"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="text-xs font-semibold text-[var(--text)]">
                          Exclusion{" "}
                          {index + 1}
                        </div>

                        <div className="mt-0.5 text-[10px] font-medium text-[var(--accent)]">
                          {
                            exclusion.category
                          }
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={() =>
                          removeManualExclusion(
                            exclusion.id
                          )
                        }
                        className="text-[10px] font-semibold text-[var(--danger)] hover:underline"
                      >
                        Remove
                      </button>
                    </div>

                    <div className="mt-2 text-[10px] leading-4 text-[var(--text-secondary)]">
                      {
                        exclusion.comment
                      }
                    </div>
                  </div>
                )
              )}
            </div>
          )}
        </div>

        {pendingExclusion && (
          <div className="absolute right-4 top-4 z-40 w-[340px] rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 shadow-lg">
            <div className="text-sm font-semibold text-[var(--text)]">
              Manual Exclusion
            </div>

            <div className="mt-1 text-xs leading-4 text-[var(--text-secondary)]">
              Add information describing why this
              area should be excluded from analysis.
            </div>

            <div className="mt-4">
              <label className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
                Category
              </label>

              <select
                value={
                  exclusionCategory
                }
                onChange={(e) =>
                  setExclusionCategory(
                    e.target.value
                  )
                }
                className="mt-1 w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-xs text-[var(--text)] focus:outline-none"
              >
                <option value="">
                  Select category
                </option>

                <option value="Development / Construction">
                  Development /
                  Construction
                </option>

                <option value="Road / Infrastructure">
                  Road /
                  Infrastructure
                </option>

                <option value="Agriculture">
                  Agriculture
                </option>

                <option value="Bare Soil">
                  Bare Soil
                </option>

                <option value="Water / Inundation">
                  Water /
                  Inundation
                </option>

                <option value="Other">
                  Other
                </option>
              </select>
            </div>

            <div className="mt-4">
              <label className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
                Comment
                <span className="ml-1 text-[var(--danger)]">
                  *
                </span>
              </label>

              <textarea
                value={
                  exclusionComment
                }
                onChange={(e) =>
                  setExclusionComment(
                    e.target.value
                  )
                }
                rows={4}
                placeholder="Describe why this area should be excluded..."
                className="mt-1 w-full resize-none rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-xs text-[var(--text)] placeholder:text-[var(--text-secondary)] focus:outline-none"
              />
            </div>

            {formError && (
              <div className="mt-3 rounded-md border border-[var(--danger)]/30 bg-[var(--danger)]/5 px-3 py-2 text-xs text-[var(--danger)]">
                {formError}
              </div>
            )}

            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={
                  cancelPendingExclusion
                }
                className="rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-xs font-semibold text-[var(--text)]"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={
                  savePendingExclusion
                }
                className="rounded-md bg-[var(--accent)] px-3 py-2 text-xs font-semibold text-white"
              >
                Save Exclusion
              </button>
            </div>
          </div>
        )}

        <div className="absolute bottom-4 left-4 right-4 z-30 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs text-[var(--text-secondary)] shadow-sm">
          {status}
        </div>
      </div>
    </section>
  );
}