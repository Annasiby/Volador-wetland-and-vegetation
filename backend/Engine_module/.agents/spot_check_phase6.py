import rasterio
import numpy as np
import json
from pathlib import Path

def main():
    pair_name = "2023-08-14_2023-08-28"
    
    # Load input rasters
    t_map_path = "outputs/transitions/" + pair_name + "/transition_map.tif"
    cva_path = "outputs/cva/" + pair_name + "/cva_magnitude.tif"
    ndvi_path = "outputs/cva/" + pair_name + "/delta_ndvi.tif"
    gndvi_path = "outputs/cva/" + pair_name + "/delta_gndvi.tif"
    ndre_path = "outputs/cva/" + pair_name + "/delta_ndre.tif"
    mask_path = "outputs/inference/" + pair_name + "/overlap_mask.tif"
    
    # Load output rasters
    deg_map_path = "outputs/degradation/degradation_map.tif"
    rec_map_path = "outputs/degradation/recovery_map.tif"
    res_map_path = "outputs/degradation/residual_stress.tif"
    canopy_map_path = "outputs/degradation/canopy_loss_map.tif"
    
    # Load default stats to get CVA threshold
    with open("outputs/degradation/degradation_statistics.json", "r") as f:
        stats = json.load(f)
    with open("outputs/degradation/degradation_metadata.json", "r") as f:
        meta = json.load(f)
        
    cva_threshold = meta["residual_stress_rule"]["threshold_value"]
    print(f"Residual CVA Threshold: {cva_threshold:.6f}")
    
    # Open all
    with (
        rasterio.open(t_map_path) as src_t,
        rasterio.open(cva_path) as src_c,
        rasterio.open(ndvi_path) as src_ndvi,
        rasterio.open(gndvi_path) as src_gndvi,
        rasterio.open(ndre_path) as src_ndre,
        rasterio.open(mask_path) as src_mask,
        rasterio.open(deg_map_path) as src_deg,
        rasterio.open(rec_map_path) as src_rec,
        rasterio.open(res_map_path) as src_res,
        rasterio.open(canopy_map_path) as src_canopy
    ):
        t_arr = src_t.read(1)
        cva_arr = src_c.read(1)
        ndvi_arr = src_ndvi.read(1)
        gndvi_arr = src_gndvi.read(1)
        ndre_arr = src_ndre.read(1)
        mask_arr = src_mask.read(1)
        
        deg_arr = src_deg.read(1)
        rec_arr = src_rec.read(1)
        res_arr = src_res.read(1)
        canopy_arr = src_canopy.read(1)
        
    # Find valid pixels
    valid_indices = np.where((mask_arr == 1) & (t_arr != 255) & (cva_arr != -9999.0))
    valid_count = len(valid_indices[0])
    print(f"Total valid pixels for selection: {valid_count}")
    
    # Let's group pixels by transition IDs to select a representative sample
    selected_pixels = []
    
    # We want to select at least 20 pixels, including as many transition IDs as possible
    # Group by transition ID
    unique_tids = np.unique(t_arr[valid_indices])
    print(f"Unique valid transition IDs present in scene: {unique_tids}")
    
    # Select 2 pixels from each transition ID if possible
    np.random.seed(42)
    for tid in unique_tids:
        idx_for_tid = np.where((mask_arr == 1) & (t_arr == tid) & (cva_arr != -9999.0))
        count = len(idx_for_tid[0])
        if count > 0:
            # pick up to 2 pixels randomly
            choice = np.random.choice(count, min(count, 2), replace=False)
            for ch in choice:
                row = idx_for_tid[0][ch]
                col = idx_for_tid[1][ch]
                selected_pixels.append((row, col, int(tid)))
                
    # If we need more pixels to reach at least 20, fill randomly
    if len(selected_pixels) < 20:
        remaining = 20 - len(selected_pixels)
        choice = np.random.choice(valid_count, remaining, replace=False)
        for ch in choice:
            row = valid_indices[0][ch]
            col = valid_indices[1][ch]
            tid = int(t_arr[row, col])
            selected_pixels.append((row, col, tid))
            
    # Print the report
    print("\n--- SPOT CHECK REPORT ---")
    print(f"{'No.':<4} | {'Row':<5} | {'Col':<5} | {'TransID':<8} | {'CVA':<8} | {'dNDVI':<8} | {'dGNDVI':<8} | {'dNDRE':<8} | {'Deg':<4} | {'Rec':<4} | {'Res':<4} | {'Can':<4} | {'Expected':<12} | {'Matches?'}")
    print("-" * 120)
    
    for i, (row, col, tid) in enumerate(selected_pixels[:25]): # limit to 25
        cva_val = float(cva_arr[row, col])
        ndvi_val = float(ndvi_arr[row, col])
        gndvi_val = float(gndvi_arr[row, col])
        ndre_val = float(ndre_arr[row, col])
        
        deg_val = int(deg_arr[row, col])
        rec_val = int(rec_arr[row, col])
        res_val = int(res_arr[row, col])
        canopy_val = int(canopy_arr[row, col])
        
        # Calculate expected based on tid and CVA
        # H->M(14), H->S(13), M->S(9) -> Deg
        # M->H(11), S->M(6), S->H(7) -> Rec
        # H->B(12), M->B(8), S->B(4) -> Canopy
        # H->H(15), M->M(10), S->S(5) -> check CVA >= threshold
        expected_deg = 1 if tid in (9, 13, 14) else 0
        expected_rec = 1 if tid in (6, 7, 11) else 0
        expected_canopy = 1 if tid in (4, 8, 12) else 0
        expected_res = 1 if (tid in (5, 10, 15) and cva_val >= cva_threshold) else 0
        
        match = (
            (deg_val == expected_deg) and
            (rec_val == expected_rec) and
            (res_val == expected_res) and
            (canopy_val == expected_canopy)
        )
        
        exp_label = []
        if expected_deg: exp_label.append("Degradation")
        if expected_rec: exp_label.append("Recovery")
        if expected_canopy: exp_label.append("CanopyLoss")
        if expected_res: exp_label.append("ResidualStress")
        if not (expected_deg or expected_rec or expected_canopy or expected_res):
            exp_label.append("NoChange")
            
        exp_str = ",".join(exp_label)
        
        print(f"{i+1:<4} | {row:<5} | {col:<5} | {tid:<8} | {cva_val:<8.4f} | {ndvi_val:<8.4f} | {gndvi_val:<8.4f} | {ndre_val:<8.4f} | {deg_val:<4} | {rec_val:<4} | {res_val:<4} | {canopy_val:<4} | {exp_str:<12} | {'PASS' if match else 'FAIL'}")

if __name__ == "__main__":
    main()
