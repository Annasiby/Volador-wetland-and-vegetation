export type DatasetValidationState =
  | "NOT_VALIDATED"
  | "VALIDATING"
  | "READY"
  | "WARNING"
  | "INVALID";

export type BandName = "green" | "red" | "rededge" | "nir";

export type DatasetMetadata = {
  date: string | null;
  crs: string | null;
  width: number | null;
  height: number | null;
  resolution: string | null;
  transform: string | null;
  nodata: string | null;
  dtype: string | null;
};

export type Dataset = {
  id: string;
  label: string;
  metadata: DatasetMetadata;
  bands: Record<BandName, string | null>;
  validation: DatasetValidationState;
};