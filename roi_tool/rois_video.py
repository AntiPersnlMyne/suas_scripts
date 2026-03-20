# File IO
import sys, os, csv
from pathlib import Path

# Vector operations
import numpy as np
from numpy.typing import NDArray

# GUI framework
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QSlider,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
)
from PyQt6.QtGui import QAction

# Visualization and ROI handling
import pyqtgraph as pg

# Save/load ROI annotations
import pickle

# Read RJPEG values
from flir import RJPEG

# Let's try this out, shall we?
from dataclasses import dataclass, replace


# -------------------------------
# ROI Data
# -------------------------------
@dataclass(frozen=False)
class ROIData:
    """Data contained for an ROI instance
    
    Attributes:
        id: ROI's "name" or identifier
        mean: RAD mean
        stdev: RAD standard deviation
        mean_display: The mean value displayed to the GUI
        stdev_display: The standard deviation value displayed to the GUI
    """
    id: int
    mean: float | None = None
    stdev: float | None = None
    mean_display: float | None = None
    stdev_display: float | None = None


# -------------------------------------------------------
# Sets of functions for a particular purpose ("Service")
# -------------------------------------------------------
@dataclass(frozen=True)
class ImageService:
    """Handles loading and calibrating thermal image data."""
    calcoeff: np.ndarray

    def load(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Reads an RJPEG image and applies radiometric calibration.

        Args:
            path (str): The file path to the RJPEG image.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - dc_img: The raw digital counts image array.
                - rad_img: The radiometrically calibrated image array.

        Raises:
            FileNotFoundError: If the specified path does not exist.
            ValueError: If the calibration coefficients array is misconfigured.
        """
        # Load proprietary FLIR format
        rjpeg = RJPEG(path)
        dc = rjpeg.raw_counts
        
        # Apply thermal calibration (slope + offset) to raw counts
        rad = dc * self.calcoeff[:, :, 0] + self.calcoeff[:, :, 1]
        
        return dc, rad


@dataclass(frozen=True)
class RegistrationServiceOld:
    """Manages spatial alignment and coordinate transformations between video frames."""
    registered_to: NDArray # variables loaded from
    matrices: NDArray      # Joe's 'register_video.py' script

    def transformation(self, from_idx: int, to_idx: int, image_shape:tuple) -> NDArray|None:
        """
        Calculates the spatial transformation matrix needed to map coordinates from one frame to another.

        Args:
            from_idx (int): The index of the source frame.
            to_idx (int): The index of the target frame.
            image_shape (tuple): The dimensions of the image (height, width, channels).

        Returns:
            np.ndarray | None: A 3x3 affine transformation matrix, or None if the frames 
            do not share a common registration anchor.
        """
        if from_idx == to_idx:
            return np.eye(3)

        # Resolve paths to a common anchor frame
        m_from, root_from = self._get_transform_to_ref(from_idx)
        m_to, root_to = self._get_transform_to_ref(to_idx)

        # Disjointed frames cannot be spatially linked
        if root_from != root_to:
            return None
        
        # Combine matrices to bridge the two frames
        m_combined = np.linalg.inv(m_to) @ m_from
        
        print(f"m_combined det: {np.linalg.det(m_combined)}")
        
        # Shift origin to center of image for correct scale/rotation
        return self._apply_centered(m_combined, image_shape)
    
    def _get_transform_to_ref(self, start_idx: int) -> tuple[np.ndarray, int]:
        """
        Traces a frame's registration path back to its refernce frame.

        Args:
            start_idx (int): The frame index to trace.

        Returns:
            tuple[np.ndarray, int]: The accumulated 3x3 transformation matrix and the root frame index.
        """
        m_total = np.eye(3)
        current = start_idx
        
        while True:
            next_idx = self.registered_to[current]
            
            # Stop condition: Frame is its own anchor or unanchored
            if next_idx == current or next_idx is None:
                break
                
            m_total = self.matrices[current] @ m_total
            current = next_idx
            
        return m_total, current

    def _apply_centered(self, transform_matrix:NDArray, image_shape:tuple) -> NDArray:
        """
        Shifts the transformation origin to the image center to preserve rotation/scaling accuracy.

        Args:
            transform_matrix (np.ndarray): The original 3x3 transformation matrix.
            shape (tuple): The dimensions of the image.

        Returns:
            np.ndarray: The center-adjusted 3x3 transformation matrix.
        """
        # Rotate around the center of the image
        h, w = image_shape[:2]
        cy, cx = h / 2, w / 2

        # Transformation matrices
        t_to_origin = np.array([
            [1, 0, -cy], 
            [0, 1, -cx], 
            [0, 0, 1],
        ])
        t_back = np.array([
            [1, 0, cy], 
            [0, 1, cx], 
            [0, 0, 1],
        ])

        return t_back @ transform_matrix @ t_to_origin

@dataclass(frozen=True)
class RegistrationService:
    """Manages spatial alignment and coordinate transformations between video frames."""
    registered_to: NDArray
    matrices: NDArray

    def transformation(self, from_idx: int, to_idx: int) -> NDArray|None:
        """
        Calculates the spatial transformation matrix needed to map coordinates from one frame to another.
        """
        if from_idx == to_idx:
            return np.eye(3)

        # Resolve paths to a common anchor frame
        m_from, root_from = self._get_transform_to_ref(from_idx)
        m_to, root_to = self._get_transform_to_ref(to_idx)

        # Disjointed frames cannot be spatially linked
        if root_from != root_to:
            return None

        # Combine matrices to bridge the two frames
        # m_from maps root -> from_idx. So inv(m_from) maps from_idx -> root.
        # m_to maps root -> to_idx.
        # Combined maps from_idx -> root -> to_idx.
        m_combined = m_to @ np.linalg.inv(m_from)

        return m_combined

    def _get_transform_to_ref(self, start_idx: int) -> tuple[np.ndarray, int]:
        """
        Traces a frame's registration path back to its reference frame.
        """
        m_total = np.eye(3)
        current = start_idx
        visited = set()

        while True:
            # Cycle detection
            if current in visited:
                print(f"Warning: Cyclic registration detected at frame {current}. Breaking loop.")
                break
            visited.add(current)

            next_idx = self.registered_to[current]

            # Stop condition: Frame is its own anchor or unanchored
            if next_idx == current or next_idx is None:
                break

            # Post-multiply since the new matrices map reference -> current
            m_total = m_total @ self.matrices[current]
            current = next_idx

        return m_total, current

@dataclass(frozen=True)
class ROITransformService:
    """Handles the geometric manipulation and extraction of Region of Interest (ROI) data."""

    def transform(self, roi: pg.PolyLineROI, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Applies an affine transformation to the boundary points of an ROI.
        """
        # Extract local coordinates and map to parent scene
        points = []
        for handle in roi.getHandles():
            pos = roi.mapToParent(handle.pos())
            # NEW WAY: ORB matrices use [x, y, 1], NOT [y, x, 1]
            points.append([pos.x(), pos.y(), 1.0])

        # Apply spatial transformation
        points = np.array(points).T
        transformed = transform_matrix @ points

        # Normalize homogenous coordinates and format as (x,y) pairs
        coords = transformed[:2] / transformed[2]

        # We no longer need to reverse the columns with [:, [1, 0]]
        return coords.T

    def extract_transform_data(self, roi: pg.PolyLineROI, transform_matrix: np.ndarray) -> dict:
        """
        Extracts boundary and metadata from an ROI after transformation.

        Args:
            roi (pg.PolyLineROI): The graphical ROI to process.
            transform_matrix (np.ndarray): The 3x3 transformation matrix.

        Returns:
            dict: A lightweight representation containing "points" (np.ndarray) and "metadata" (ROIData).
        """
        new_coords = self.transform(roi, transform_matrix)
        
        fresh_metadata = replace(roi.metadata, mean=None, stdev=None, mean_display=None, stdev_display=None)
        
        return {
            "points": new_coords,
            "metadata": fresh_metadata
        }

@dataclass(frozen=True)
class ROITransformServiceOld:
    """Handles the geometric manipulation and extraction of Region of Interest (ROI) data."""
    
    def transform(self, roi: pg.PolyLineROI, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Applies an affine transformation to the boundary points of an ROI.

        Args:
            roi (pg.PolyLineROI): The graphical ROI object to transform.
            transform_matrix (np.ndarray): The 3x3 transformation matrix to apply.

        Returns:
            np.ndarray: An Nx2 array of the newly transformed (X, Y) coordinates.
        """
        # Extract local coordinates and map to parent scene
        points = []
        for handle in roi.getHandles():
            pos = roi.mapToParent(handle.pos())
            points.append([pos.y(), pos.x(), 1.0])

        # Apply spatial transformation
        points = np.array(points).T
        transformed = transform_matrix @ points
        
        # Normalize homogenous coordinates and format as (x,y) pairs
        coords = transformed[:2] / transformed[2]
        return coords.T[:, [1, 0]]
    
    def extract_transform_data(self, roi: pg.PolyLineROI, transform_matrix: np.ndarray) -> dict:
        """
        Extracts boundary and metadata from an ROI after transformation.

        Args:
            roi (pg.PolyLineROI): The graphical ROI to process.
            transform_matrix (np.ndarray): The 3x3 transformation matrix.

        Returns:
            dict: A lightweight representation containing "points" (np.ndarray) and "metadata" (ROIData).
        """
        new_coords = self.transform(roi, transform_matrix)
        
        fresh_metadata = replace(roi.metadata, mean=None, stdev=None, mean_display=None, stdev_display=None)
        
        return {
            "points": new_coords,
            "metadata": fresh_metadata
        }


@dataclass(frozen=True)
class ROIStatsService:
    """Calculates statistical metrics for regions within thermal images."""    
    
    def compute_stats(self, roi: pg.PolyLineROI, image: NDArray, image_item: pg.ImageItem = None) -> tuple[float, float]:
        """
        Calculates the mean and standard deviation of pixel values within an ROI.

        Args:
            roi (pg.PolyLineROI): The region of interest.
            image (np.ndarray): The underlying 2D image data.
            image_item (pg.ImageItem, optional): The PyQtGraph ImageItem for mapping context.

        Returns:
            tuple[float, float]: The (mean, standard deviation) of the bounded region. 
            Returns (nan, nan) if the ROI is detached or empty.
        """
        if roi.scene() is None:
            return float("nan"), float("nan")

        # Map ROI boundaries strictly to the underlying data array
        region = roi.getArrayRegion(image, image_item, axes=(1, 0))
        
        mask = roi.getArrayRegion(np.ones_like(image, dtype=np.uint8), image_item, axes=(1, 0))
        
        # print(f"Mask size: {mask.shape}")

        # Filter pixels fully enclosed by the ROI
        valid = mask > 0
        if np.any(valid):
            values = region[valid]
            return float(np.mean(values)), float(np.std(values))

        return float("nan"), float("nan")


@dataclass(frozen=True)
class ExportService:
    save_file: str
    savedir: str

    def save_pickle(self, roi_data: dict, total_frames: int) -> None:
        """
        Serializes current ROI geometries and metadata to a binary file.

        Args:
            roi_data (dict): Mapping of frame indices to lists of ROIs (or lightweight dicts).
            total_frames (int): Total number of frames in the active sequence.
            
        Raises:
            IOError: If the destination path is unwritable.
        """
        rois_save = []
        for i in range(total_frames):
            frame_rois = []
            if i in roi_data:
                for roi in roi_data[i]:
                    # Standardize data format whether it is a GUI widget or a dict
                    if isinstance(roi, dict):
                        meta = roi["metadata"]
                        points = roi["points"]
                    else:
                        meta = roi.metadata
                        points = [[roi.mapToParent(h.pos()).x(), roi.mapToParent(h.pos()).y()] for h in roi.getHandles()]
                    
                    # Store metadata into pickle-albe dict
                    meta_dict = {
                        "id": meta.id,
                        "mean": meta.mean,
                        "stdev": meta.stdev,
                        "mean_display": meta.mean_display,
                        "stdev_display": meta.stdev_display
                    }
                    frame_rois.append({"metadata": meta_dict, "points": points})
            rois_save.append(frame_rois)

        # Write information to pickle
        with open(self.save_file, "wb") as f:
            pickle.dump(rois_save, f)


    def export_csv(self, roi_data: dict) -> None:
        """
        Exports computed ROI statistics (mean, stdev) to a CSV file.

        Args:
            roi_data (dict): Mapping of frame indices to lists of ROIs.
            
        Raises:
            IOError: If the CSV file cannot be created or written to.
        """
        csv_path = str(Path(self.savedir, "roi_stats.csv"))
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "instance", "mean", "stdev"])
            
            for frame_idx, rois in roi_data.items():
                for roi in rois:
                    meta = roi["metadata"] if isinstance(roi, dict) else roi.metadata
                    writer.writerow([frame_idx, meta.id, meta.mean, meta.stdev])


# -------------------------------
# GUI Classes
# -------------------------------
class LabeledPolyLineROI(pg.PolyLineROI):
    def __init__(
        self,
        positions,
        label_text,
        label_color=None,
        label_offset=(0, -10),
        closed=False,
        **kwargs,
    ):
        """
        A PolyLineROI with an attached numeric/text label.

        Args:
            positions:     List of (x, y) positions for the polyline handles.
            label_text:    Text or number to display.
            label_color:   Color for the label (defaults to match the ROI pen).
            label_offset:  (dx, dy) pixel offset from the label anchor point.
            closed:        Whether the polyline is closed.
            **kwargs:      Passed to PolyLineROI (e.g. pen='r').
        """
        super().__init__(positions, closed=closed, **kwargs)

        # Infer color from pen if not specified
        if label_color is None:
            pen = kwargs.get("pen", "w")
            label_color = pen if isinstance(pen, str) else "w"

        self._label_offset = label_offset

        self.label = pg.TextItem(
            text=str(label_text),
            color=label_color,
            anchor=(0.5, 0.5),
        )

        self.sigRegionChangeFinished.connect(self._update_label)

    def _get_centroid(self):
        """Compute the centroid of all handle positions in scene coordinates."""
        handles = self.getHandles()
        if not handles:
            return QtCore.QPointF(0, 0)

        scene_positions = [h.scenePos() for h in handles]
        cx = sum(p.x() for p in scene_positions) / len(scene_positions)
        cy = sum(p.y() for p in scene_positions) / len(scene_positions)
        return QtCore.QPointF(cx, cy)

    def _update_label(self):
        centroid = self._get_centroid()

        # Convert scene coords to the label's parent (ViewBox) coords
        if self.label.scene() is not None:
            view = self.label.parentItem()
            if view is not None:
                local = view.mapFromScene(centroid)
                self.label.setPos(
                    local.x() + self._label_offset[0],
                    local.y() + self._label_offset[1],
                )
            else:
                self.label.setPos(
                    centroid.x() + self._label_offset[0],
                    centroid.y() + self._label_offset[1],
                )

    def addToView(self, view):
        """Add both the ROI and its label to a ViewBox."""
        view.addItem(self)
        view.addItem(self.label)
        self._update_label()

    def removeFromView(self, view):
        """Clean removal of both the ROI and its label."""
        view.removeItem(self.label)
        view.removeItem(self)

    def setLabel(self, text):
        """Update the label text after creation."""
        self.label.setText(str(text))

    def setLabelColor(self, color):
        """Update the label color after creation."""
        self.label.setColor(color)


class ImageViewer(QMainWindow):
    def __init__(
        self,
        save_dir,
        registration_path,
        calibration_coefficients_path,
    ):
        super().__init__()

        self.savedir = save_dir
        self.registrationpath = registration_path
        self.calibration_coefficients_path = calibration_coefficients_path

        # Load data
        self.load_all_data()

        # Sets of functions ("Services") 
        self.image_service = ImageService(self.calcoeff)
        self.registration = RegistrationService(
            registered_to=self.registered_to,
            matrices=self.transform_matrices,
        )
        self.transform_service = ROITransformService()
        self.stats_service = ROIStatsService()
        self.export_service = ExportService(self.save_file, save_dir)

        # Build UI
        self.build_ui()

        # Load First Frame
        self.update_frame(self.current_index)
        self._auto_scale_histogram(self.current_img_dc)
        

    # --------------------------------------------------------
    # (Private) UI Display Helpers
    # --------------------------------------------------------

    def _hide_current_rois(self):
        """Remove ROIs belonging to the previous frame from the view."""
        if self.current_img is None:
            return

        for roi in self.roi_data.get(self.current_index, []):
            roi.removeFromView(self.view_lwir)

    def _refresh_current_rois(self):
        """
        Forces a recalculation of ROI statistics and label updates 
        for all ROIs in the currently active frame.
        """
        if self.current_index not in self.roi_data:
            return
        
        current_rois = self.roi_data.get(self.current_index, [])
        for roi in current_rois:
            self._on_roi_changed(roi)
            
    def _load_frame(self, index: int):
        """Load image data and update current image state"""
        
        # Load RJPEG files
        path = self.files[index]
        dc_img, rad_img = self.image_service.load(path)

        # Set current DC and RAD
        self.current_img_dc = dc_img
        self.current_img_rad = rad_img

        # Keep displayed image as DC 
        self.current_img = dc_img
        self.current_index = index

        # Update PyQt window header information
        self.label.setText(
            f"Is Reference: {index in self.reference_frames}, Registered To Frame #{self.registered_to[index]}, "
            f"File: {os.path.basename(path)}"
        )

    def _ensure_rois_exist_for_frame(self, index: int) -> None:
        """
        If navigating forward into a frame with no ROIs,
        auto-clone from its registered reference frame.
        """

        if index in self.roi_data:
            return

        ref_idx = self.registration.registered_to[index]
        if ref_idx not in self.roi_data:
            return

        M = self.registration.transformation(ref_idx, index)

        if M is None:
            return

        new_rois = []
        for roi in self.roi_data[ref_idx]:
            new_roi = self.transform_service.extract_transform_data(roi, M)
            new_rois.append(new_roi)

        self.roi_data[index] = new_rois

    def _display_current_image(self):
        """Push current image to ImageView (what user sees)"""
        self.view_lwir.setImage(
            self.current_img,
            axes={"x": 1, "y": 0},
            autoHistogramRange=False,
            autoLevels=False,
            autoRange=True,
        )

    def _show_rois_for_frame(self, index: int) -> None:
        """Display the ROIs for the given frame
        
        Args:
            index (int): Current frame index
        """
        
        instantiated_rois = []
        for item in self.roi_data.get(index, []):
            if isinstance(item, dict):
                roi = LabeledPolyLineROI(item["points"], item["metadata"].id, closed=True)
                roi.metadata = item["metadata"]
                instantiated_rois.append(roi)
            else:
                instantiated_rois.append(item)
                
        if index in self.roi_data:
            self.roi_data[index] = instantiated_rois
        
        for roi in self.roi_data.get(index, []):
            roi.addToView(self.view_lwir)
            try:
                roi.sigRegionChangeFinished.disconnect()
            except (TypeError, RuntimeError):
                pass
            roi.sigRegionChangeFinished.connect(self._on_roi_changed)
            self._on_roi_changed(roi)

    def _on_roi_changed(self, roi:pg.PolyLineROI):
        """Triggered when ROI geometry changes or stats toggle"""
        # Set which stats (RAD vs DC) is being displayed
        stats_source_image = self.current_img_rad if self.display_calibrated_values else self.current_img_dc
        
        if stats_source_image is None:
            return

        # Re-calculate stats for the current frame's image
        mean, std = self.stats_service.compute_stats(
            roi=roi, 
            image=stats_source_image, 
            image_item=self.view_lwir.imageItem,
        )

        # Add mean, std to ROI metadata properties
        roi.metadata.mean_display = mean
        roi.metadata.stdev_display = std

        # Update the stats on the GUI display
        self._update_roi_display(roi)

    def _update_roi_display(self, roi:pg.PolyLineROI):
        """Update label text based on display mode."""

        mean = roi.metadata.mean_display
        std = roi.metadata.stdev_display

        # Display only the ID if no stats possible
        if mean is None or std is None:
            text = f"{roi.metadata.id}"
        
        # Display stats alongside ROI's ID
        else:
            # RAD for calibration mode
            # DC for raw digital count
            prefix = "RAD" if self.display_calibrated_values else "DC"
            
            # Set actual ROI text
            text = (
                f"{roi.metadata.id}\n"
                f"{prefix} µ={mean:.1f}\n"
                f"{prefix} σ={std:.1f}"
            )

        roi.setLabel(text)

    def _clear_frame_rois(self, frame_index: int):
        """Remove ROIs from a frame if it's not the frame it belongs to"""

        # Check if frame hasn't been annotated before
        if frame_index not in self.roi_data: return
        if frame_index == self.current_index: return

        # Remove irrelevant ROIs to this frame
        for roi in self.roi_data[frame_index]:
            # Check if it's an instantiated widget before attempting to remove from view
            if not isinstance(roi, dict):
                roi.removeFromView(self.view_lwir)


    # --------------------------------------------------------
    # (Private) ROI Helpers
    # --------------------------------------------------------

    def _create_default_roi(self):
        """Create a default square ROI with new ID. Placed the ROI in top left."""

        points = [
            [10, 10],
            [10, 60],
            [60, 60],
            [60, 10],
        ]

        roi = LabeledPolyLineROI(points, self.next_roi_id, closed=True)

        roi.metadata = ROIData(id=self.next_roi_id)

        self.next_roi_id += 1

        return roi

    def _register_new_roi(self, roi:pg.PolyLineROI):
        """Store ROI in internal data structure."""
        self.roi_data.setdefault(self.current_index, []).append(roi)

    def _attach_roi_to_view(self, roi:pg.PolyLineROI):
        """Add ROI to view and connect change handler."""

        roi.addToView(self.view_lwir)

        roi.sigRegionChangeFinished.connect(lambda _, r=roi: self._on_roi_changed(r))

        # Immediately compute stats
        self._on_roi_changed(roi)

    def _on_save_stats_clicked(self) -> None:
        """Prompts the user before computing and saving statistics."""
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Save Stats",
            "Compute and export statistics for all ROIs across all frames?\nThis may take a moment.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        # Proceed if the user clicked Yes
        if reply == QMessageBox.StandardButton.Yes:
            # Update ROIs, save stats to CSV, save ROIs to pickle
            self._save_stats_and_rois()
            
            # Show completion dialog
            QMessageBox.information(
                self,
                "Save Stats Completed",
                "Successfully computed and saved ROI statistics"
            )

    def _update_all_roi_statistics(self):
        """Recompute calibrated statistics for all ROIs on all frames"""
        
        # Create a single reusable temporary ROI
        temp_roi = None
        
        for frame_idx, rois in self.roi_data.items():
            if not rois:
                continue

            path = self.files[frame_idx]
            _, img = self.image_service.load(path)

            for roi_item in rois:
                is_dict = isinstance(roi_item, dict)

                if is_dict:
                    # Reuse temp_roi
                    if temp_roi is None:
                        temp_roi = LabeledPolyLineROI([], 0, closed=True)
                    
                    # Update points in-place
                    temp_roi.setPoints(roi_item["points"])
                    temp_roi.metadata = roi_item["metadata"]
                    
                    # Add to view once per ROI batch 
                    if temp_roi.scene() is None:
                        temp_roi.addToView(self.view_lwir)
                    
                    mean, std = self.stats_service.compute_stats(temp_roi, img, self.view_lwir.imageItem)
                    
                    roi_item["metadata"].mean = mean
                    roi_item["metadata"].stdev = std
                else:
                    # For GUI widgets: minimize view operations
                    was_in_view = roi_item.scene() is not None
                    
                    if not was_in_view:
                        roi_item.addToView(self.view_lwir)
                    
                    mean, std = self.stats_service.compute_stats(roi_item, img, self.view_lwir.imageItem)
                    
                    if not was_in_view:
                        roi_item.removeFromView(self.view_lwir)
                    
                    roi_item.metadata.mean = mean
                    roi_item.metadata.stdev = std
        
        # Clean up temporary ROI after all stats are computed
        if temp_roi is not None and temp_roi.scene() is not None:
            temp_roi.removeFromView(self.view_lwir)

    def _toggle_stats(self):
        """
        Toggles display between DC and Calibrated values and refreshes labels.
        """
        self.display_calibrated_values = not self.display_calibrated_values
        
        # Update button text to reflect next state
        button_label = "Toggle Values (RAD)" if not self.display_calibrated_values else "Toggle Values (DC)"
        self.toggle_values_button.setText(button_label)

        # Refresh the statistics for visible ROIs 
        self._refresh_current_rois()
            
    def _auto_scale_histogram(self, image: NDArray) -> None:
        """
        Calculates and sets the histogram display levels to mean +/- 1 standard deviation.

        Args:
            image (NDArray): The 2D thermal image array currently being displayed.
        """
        # Calculate image-wide statistics
        mean_val = float(np.mean(image))
        std_val = float(np.std(image))

        # Define optimal viewing bounds (1 std dev)
        vmin = mean_val - std_val
        vmax = mean_val + std_val

        # Snap the histogram handles to the new bounds
        self.view_lwir.imageItem.setLevels([vmin, vmax])
            
    # --------------------------------------------------------
    # (Private) Propogation Helpers
    # --------------------------------------------------------

    def _can_propagate(self) -> bool:
        """Return True if propagation is possible."""

        # Warn user no ROIs in current frame -> can't propogate them
        has_rois = (
            self.current_index in self.roi_data
            and len(self.roi_data[self.current_index]) > 0
        )
        if not has_rois:
            QMessageBox.warning(
                self, "Cannot Propagate", "No ROIs in current frame to propagate."
            )
            return False

        # Can't propogate if no future frames to be propogated to
        has_future_frames = self.current_index < len(self.files) - 1
        if not has_future_frames:
            return False

        # Allow propogation if checks pass
        return True

    def _confirm_propagation(self) -> bool:
        """Ask user for confirmation before propagation."""

        # Define start and stop region of propogation
        roi_count = len(self.roi_data[self.current_index])
        start = self.current_index + 1
        end = len(self.files) - 1

        # Message to user asking for confirmation
        reply = QMessageBox.question(
            self,
            "Confirm Propagation",
            f"Propagate {roi_count} ROIs "
            f"from frame {self.current_index} "
            f"to frames {start}-{end}?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        return reply == QMessageBox.StandardButton.Yes

    def _propagate_from_current_frame(self) -> int:
        """Propagate ROIs forward using Joe's registration service"""

        # Index of starting frame of propogation
        source_idx = self.current_index
        
        # ROIs to propogate down
        source_rois = self.roi_data[source_idx]

        # Iterate from current frame to end frame
        propagated_count = 0
        for target_idx in range(source_idx + 1, len(self.files)):

            # Transformation matrix to register
            M = self.registration.transformation(
                source_idx, target_idx # , self.current_img.shape
            )

            if M is None: continue

            new_rois_data = [
                self.transform_service.extract_transform_data(roi, M) for roi in source_rois
            ]

            # Add ROI to ROI list
            self.roi_data[target_idx] = new_rois_data
            propagated_count += 1

        return propagated_count

    def _show_propagation_result(self, count: int):
        """Display propogation success to user in GUI"""
        QMessageBox.information(
            self,
            "Propagation Complete",
            f"Successfully propagated ROIs to {count} frames.",
        )

    # --------------------------------------------------------
    # (Private) Data Loading Helpers
    # --------------------------------------------------------

    def _load_calibration(self) -> None:
        """Load calibration coefficients from NumPy file"""

        if not Path(self.calibration_coefficients_path).exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_coefficients_path}")

        self.calcoeff = np.load(self.calibration_coefficients_path)

        if self.calcoeff.ndim != 3 or self.calcoeff.shape[2] != 2:
            raise ValueError("Calibration array must have shape (H, W, 2)")

    def _load_registration(self) -> None:
        """Load frame registration metadata, file paths, and flight metadata"""
        
        if not os.path.exists(self.registrationpath):
            raise FileNotFoundError(f"Registration file not found: {self.registrationpath}")

        # Load Joe's magical registration logic (now with files and metadata included)
        registration_file = np.load(self.registrationpath)

        try:
            self.registered_to = registration_file["registered_to"]
            self.transform_matrices = registration_file["transform_matrices"]
            self.files = registration_file["files"]
            self.metadata = registration_file["metadata"]
        except KeyError as e:
            raise KeyError(f"Missing registration key: {e}")

        self.reference_frames = np.unique(self.registered_to)

    def _load_roi_metadata(self) -> None:
        """Load saved ROI data (if it exists)"""
        
        # Save ROI data to rois.pkl
        self.save_file = os.path.join(self.savedir, "rois.pkl")
        self.roi_data = {}

        # Instantiate the pickle object
        if os.path.exists(self.save_file):
            with open(self.save_file, "rb") as f:
                save_data = pickle.load(f)
                
            # Iterate through list-of-lists 
            for idx, frame_rois in enumerate(save_data):
                for roi_dict in frame_rois:
                    old_meta = roi_dict["metadata"]
                    
                    # Convert dict into Dataclass object (new method)
                    # Dict is Joe's old way, retained for backwards compatability
                    new_meta = ROIData(
                        id=old_meta["id"],
                        mean=old_meta.get("mean"),
                        stdev=old_meta.get("stdev"),
                        mean_display=old_meta.get("mean_display"),
                        stdev_display=old_meta.get("stdev_display")
                    )
                    
                    # Create ROIs from pickle data
                    # Also load their metadata
                    new_roi = LabeledPolyLineROI(roi_dict["points"], new_meta.id, closed=True)
                    new_roi.metadata = new_meta
                    
                    self.roi_data.setdefault(idx, []).append(new_roi)

    def _initialize_runtime_state(self) -> None:
        """Initialize non-persistent runtime variables."""

        # Default parameters
        self.current_index = self.registered_to[0]
        self.current_img = None
        self.current_img_dc = None
        self.current_img_rad = None
        self.display_calibrated_values = False

        # Default ID: start at 0
        max_id = -1 
        
        # Compute next ROI ID from highest ID found in pickle
        for rois in self.roi_data.values():
            for roi in rois:
                if hasattr(roi, "metadata") and roi.metadata:
                    max_id = max(max_id, roi.metadata.id)

        # Increment ID
        self.next_roi_id = max_id + 1

    def _save_stats_and_rois(self) -> None:
        """Saving ROI data and statistics."""
        
        # Update the statistics that'll be saved
        self._update_all_roi_statistics()

        # Save ROI data to pickle
        self.export_service.save_pickle(self.roi_data, len(self.files))
        
        # Save stats to CSV
        self.export_service.export_csv(self.roi_data)
        
    # ----------------
    # Main executables
    # ----------------
    def update_frame(self, index: int):
        """Updated frame after the user interfaces with GUI"""

        # Hide previous ROIs when going through video
        self._hide_current_rois()
        # Load next frame of video
        self._load_frame(index)
        # Apply Joe's registration to frames without existing ROIs 
        self._ensure_rois_exist_for_frame(index)
        # Display ROIs and stats to user
        self._display_current_image()
        self._show_rois_for_frame(index)
        

    def propagate_rois(self):
        """ROI propagation pipeline"""

        # Check if propogation is possible
        if not self._can_propagate():
            return

        # Ask user for confirmation
        if not self._confirm_propagation():
            return

        # Apply propogation
        propagated_count = self._propagate_from_current_frame()

        # Display result
        self._show_propagation_result(propagated_count)

        # Refresh current frame display
        self.update_frame(self.current_index)

    def add_roi(self):
        """Create a default ROI in top-left corner (Joe's default)"""
        roi = self._create_default_roi()
        self._register_new_roi(roi)
        self._attach_roi_to_view(roi)
        self._update_roi_display(roi)

    def load_all_data(self):
        """Data-loading pipeline."""
        # Registration, metadata, and files loading
        self._load_registration()
        # Calibration coefficients
        self._load_calibration()
        # Metadata of each ROI
        self._load_roi_metadata()
        # Get the GUI started
        self._initialize_runtime_state()

    def build_ui(self):
        """Construct all Qt widgets, layouts, and signal connections."""

        # Set PyQt GUI window title
        self.setWindowTitle("ROI Draw")

        # ========================
        # Central Container
        # ========================
        container = QWidget()
        self.setCentralWidget(container)

        main_layout = QVBoxLayout(container)

        # ========================
        # File / Frame Label
        # ========================
        self.label = QLabel("File:")
        main_layout.addWidget(self.label)

        # ========================
        # Image Viewer
        # ========================
        self.view_lwir = pg.ImageView()
        main_layout.addWidget(self.view_lwir)

        # ========================
        # Frame Slider
        # ========================
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(self.files) - 1)
        self.slider.setValue(self.current_index)
        self.slider.valueChanged.connect(self.update_frame)
        main_layout.addWidget(self.slider)

        # ========================
        # Buttons
        # ========================
        button_layout = QHBoxLayout()
        self.prev_button = QPushButton("<")
        self.next_button = QPushButton(">")
        self.add_roi_button = QPushButton("Add ROI")
        self.add_view_roi_button = QPushButton("Add ROI (in view)")
        self.save_button = QPushButton("Save Stats")
        self.toggle_values_button = QPushButton("Toggle Values (DC)")

        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.add_roi_button)
        button_layout.addWidget(self.add_view_roi_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.toggle_values_button)

        main_layout.addLayout(button_layout)

        # ========================
        # Button Connections
        # ========================
        self.prev_button.clicked.connect(
            lambda: self.slider.setValue(self.slider.value() - 1)
        )

        self.next_button.clicked.connect(
            lambda: self.slider.setValue(self.slider.value() + 1)
        )

        self.add_roi_button.clicked.connect(self.add_roi)
        self.add_view_roi_button.clicked.connect(self.add_view_roi)
        self.save_button.clicked.connect(self._on_save_stats_clicked)
        self.toggle_values_button.clicked.connect(self._toggle_stats)

        # ========================
        # Right-Click Context Menu
        # ========================
        
        # Get the viewbox (the actual 2D plotting area)
        view_box = self.view_lwir.getView()
        
        # Add context menu action/button to the menu
        self.propagate_action = QAction("Propagate ROIs", self)
        self.propagate_action.triggered.connect(self.propagate_rois)

        # Inject the action into the context menu
        view_box.menu.addAction(self.propagate_action)

    def add_view_roi(self):
        """
        Creates a new ROI that covers the central 1/4 of the current view area
        """
        # Get current view boundaries
        view_box = self.view_lwir.getView()
        rect = view_box.viewRect()
        
        # Calculate center and offset 
        center = rect.center()
        w_quarter = rect.width() / 4
        h_quarter = rect.height() / 4
        w_offset = w_quarter / 2
        h_offset = h_quarter / 2
        
        # Define 4 points centered on current view
        points = [
            [center.x() - w_offset, center.y() - h_offset],
            [center.x() - w_offset, center.y() + h_offset],
            [center.x() + w_offset, center.y() + h_offset],
            [center.x() + w_offset, center.y() - h_offset],
        ]

        # Define ROI, and add metadata to it
        roi = LabeledPolyLineROI(points, self.next_roi_id, closed=True)
        roi.metadata = ROIData(id=self.next_roi_id)
        self.next_roi_id += 1

        # Add ROI to GUI
        self._register_new_roi(roi)
        self._attach_roi_to_view(roi)
        self._update_roi_display(roi)


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Argument parsing
    args = sys.argv[1:]
    assert len(args) == 3, "Usage: python rois_video.py <save_dir> <registration_path> <calibration_coefficients_path>"
    (
        save_dir,
        registration_path,
        calibration_coefficients_path,
    ) = args
    
    # Start GUI application
    app = QApplication(sys.argv)
    window = ImageViewer(
        save_dir,
        registration_path,
        calibration_coefficients_path,
    )
    window.show()
    
    # End GUI application
    sys.exit(app.exec())
