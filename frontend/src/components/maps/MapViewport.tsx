import { useEffect, useRef, useState } from "react";
import Map from "ol/Map";
import View from "ol/View";
import TileLayer from "ol/layer/Tile";
import ImageLayer from "ol/layer/Image";
import OSM from "ol/source/OSM";
import ImageStatic from "ol/source/ImageStatic";
import { fromLonLat, transformExtent } from "ol/proj";
import { useProject } from "../../context/useProject";
import { getLayerImageUrl } from "../../api/client";
import "ol/ol.css";

export default function MapViewport() {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<Map | null>(null);
  const imageLayerRef = useRef<ImageLayer<ImageStatic> | null>(null);

  const {
    runStatus,
    layers,
    activeLayerId,
    setActiveLayerId,
  } = useProject();

  const [opacity, setOpacity] = useState<number>(0.85);

  const activeLayer =
    layers.find((l) => l.id === activeLayerId) || layers[0];

  // Initialize OpenLayers map
  useEffect(() => {
    if (!mapRef.current) return;

    const baseTileLayer = new TileLayer({
      source: new OSM(),
    });

    const initialMap = new Map({
      target: mapRef.current,
      layers: [baseTileLayer],
      view: new View({
        center: fromLonLat([13.4, 52.5]),
        zoom: 12,
      }),
    });

    mapInstanceRef.current = initialMap;

    return () => {
      initialMap.setTarget(undefined);
    };
  }, []);

  // Update raster overlay layer when active layer or run changes
  useEffect(() => {
    const map = mapInstanceRef.current;

    if (
      !map ||
      !runStatus?.run_id ||
      !activeLayer ||
      !activeLayer.ready
    ) {
      return;
    }

    // Remove existing image overlay
    if (imageLayerRef.current) {
      map.removeLayer(imageLayerRef.current);
      imageLayerRef.current = null;
    }

    // Bounds in EPSG:4326
    const rawBounds = activeLayer.bounds || [
      13.3,
      52.4,
      13.5,
      52.6,
    ];

    const imageExtent = transformExtent(
      rawBounds,
      "EPSG:4326",
      "EPSG:3857"
    );

    const imgUrl = getLayerImageUrl(
      runStatus.run_id,
      activeLayer.id
    );

    const imageSource = new ImageStatic({
      url: imgUrl,
      imageExtent,
      projection: "EPSG:3857",
    });

    const newImageLayer = new ImageLayer({
      source: imageSource,
      opacity,
    });

    map.addLayer(newImageLayer);
    imageLayerRef.current = newImageLayer;

    // Fit map view to raster extent
    map.getView().fit(imageExtent, {
      padding: [40, 40, 40, 40],
      duration: 500,
    });
  }, [
    runStatus?.run_id,
    activeLayerId,
    activeLayer,
    layers,
    opacity,
  ]);

  // Handle opacity slider change
  const handleOpacityChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const val = parseFloat(e.target.value);

    setOpacity(val);

    if (imageLayerRef.current) {
      imageLayerRef.current.setOpacity(val);
    }
  };

  return (
    <div className="relative flex min-h-[420px] flex-1 overflow-hidden rounded-lg border border-[var(--border)] bg-[#ECECE8]">
      {/* Map Container */}
      <div
        ref={mapRef}
        className="absolute inset-0 h-full w-full"
      />

      {/* Layer Control Panel */}
      <div className="absolute left-4 top-4 z-10 flex flex-col gap-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3.5 shadow-md">
        <div>
          <label className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            GIS Layer
          </label>

          <select
            value={activeLayerId}
            onChange={(e) =>
              setActiveLayerId(e.target.value)
            }
            className="mt-1 w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-1.5 text-xs font-medium text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
          >
            {layers.length > 0 ? (
              layers.map((layer) => (
                <option
                  key={layer.id}
                  value={layer.id}
                  disabled={!layer.ready}
                >
                  {layer.name}
                  {!layer.ready ? " (Not ready)" : ""}
                </option>
              ))
            ) : (
              <option value="">
                No layers loaded
              </option>
            )}
          </select>
        </div>

        {/* Opacity Control */}
        <div>
          <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            <span>Opacity</span>
            <span>{Math.round(opacity * 100)}%</span>
          </div>

          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={handleOpacityChange}
            className="mt-1 w-full accent-[var(--accent)]"
          />
        </div>
      </div>

      {/* Map Color Legend */}
      <div className="absolute bottom-4 right-4 z-10 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 shadow-md">
        <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
          {activeLayer?.name || "Legend"}
        </div>

        <div className="mt-2 flex flex-col gap-1 text-xs">
          {activeLayerId.includes("health") && (
            <>
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-[#00AA00]" />
                <span className="text-[var(--text)]">
                  Healthy Vegetation
                </span>
              </div>

              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-[#FFA500]" />
                <span className="text-[var(--text)]">
                  Moderate Stress
                </span>
              </div>

              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-[#FF0000]" />
                <span className="text-[var(--text)]">
                  Severe Stress
                </span>
              </div>

              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-[#808080]" />
                <span className="text-[var(--text)]">
                  Background
                </span>
              </div>
            </>
          )}

          {activeLayerId.includes("degradation") && (
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-sm bg-[#FF4500]" />
              <span className="text-[var(--text)]">
                Degraded Area
              </span>
            </div>
          )}

          {activeLayerId.includes("recovery") && (
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-sm bg-[#32CD32]" />
              <span className="text-[var(--text)]">
                Recovered Area
              </span>
            </div>
          )}

          {activeLayerId.includes("water") && (
            <div className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-sm bg-[#0064FF]" />
              <span className="text-[var(--text)]">
                Water Extent / Change
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}