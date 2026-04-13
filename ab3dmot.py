import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class CVKalmanTracker:
    def __init__(self, config, track_id):
        self.dt = config.get("dt", 0.1)
        self.kf = self.create_kf_cv(
            config.get("initial_state", np.zeros(6)),
            config.get("initial_covariance", 100),
        )
        self.id = track_id
        self.vehicle_type = None
        self.confidence = 0.0
        self.dimensions = [0, 0, 0]
        self.heading = 0.0
        self.age = 0
        self.hits = 0
        self.missed = 0
        self.association_history = []
        self.state = "tentative"
        self.confirmation_frames_needed = config.get("confirmation_frames_needed", 3)
        self.confirmation_window = config.get("confirmation_window", 5)
        self.deletion_missed_threshold = config.get("deletion_missed_threshold", 3)
        self.deletion_window = config.get("deletion_window", 5)

    def create_kf_cv(self, initial_state, initial_covariance):
        """Creates a Kalman Filter for a Constant Velocity (CV) model."""
        kf = KalmanFilter(dim_x=6, dim_z=3)
        dt = self.dt
        # State transition matrix F
        kf.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        # Measurement function H
        kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        kf.R = np.eye(3) * 0.1  # Measurement noise
        kf.Q = np.eye(6) * 0.01  # Process noise
        kf.x = initial_state.copy()
        kf.P = np.eye(6) * initial_covariance
        return kf

    def predict(self, dt=None):
        """Predict the next state with a given time step."""
        if dt is not None:
            self.dt = dt
            self.kf.F[:3, 3:] = np.eye(3) * dt

        self.kf.predict()
        self.age += 1

        # vx, vy = self.kf.x[3], self.kf.x[4]
        # if vx == 0 and vy == 0:
        #     self.heading = self.heading if self.heading is not None else 0.0
        # else:
        #     # Use standard arctan2(vy, vx) for heading
        #     angle = np.degrees(np.arctan2(vy, vx))
        #     if angle < 0:
        #         angle += 360
        #     self.heading = angle

    def update(self, z, dimensions, confidence, heading=None, vehicle_type=None):
        """Update the state based on a new measurement."""
        self.kf.update(z)
        self.dimensions = dimensions
        self.confidence = confidence
        if heading is not None:
            self.heading = heading
        if vehicle_type is not None:
            self.vehicle_type = vehicle_type

        self.hits += 1
        self.missed = 0
        self.association_history.append(1)

        if len(self.association_history) > self.confirmation_window:
            self.association_history.pop(0)

        if (
            self.state == "tentative"
            and self.association_history.count(1) >= self.confirmation_frames_needed
        ):
            self.state = "confirmed"
        elif self.state == "confirmed":
            if (
                self.association_history[-self.deletion_window :].count(0)
                >= self.deletion_missed_threshold
            ):
                self.state = "deleted"


class MultiObjectTracker:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        self.trackers = []
        self.next_id = 0
        self.cost_threshold = config.get("cost_threshold", 4.0)

    def predict(self, dt):
        """
        Predict all trackers.

        Args:
            dt: Time delta since last update

        Returns:
            List of active tracks.
        """

        # Predict
        for tracker in self.trackers:
            tracker.predict(dt)
        
        return self.get_tracks()

    def update(self, detections, dt):
        """
        Updates all trackers with new measurements.

        Args:
            detections: List of detection dicts or objects.
                        Expected format: {'position': [x,y,z], 'dimensions': [l,w,h], 'score': float, 'label': int/str, 'heading': float}
            dt: Time delta since last update

        Returns:
            List of active tracks.
        """

        # Associate
        cost_matrix = self.compute_cost_matrix(detections)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_detections = set()

        for t, d in zip(row_ind, col_ind):
            if cost_matrix[t, d] <= self.cost_threshold:
                detection = detections[d]
                tracker = self.trackers[t]

                # Extract detection data
                pos = detection["position"]
                dims = detection["dimensions"]
                score = detection["score"]
                heading = detection.get("heading", 0.0)
                label = detection.get("label", 0)

                tracker.update(pos, dims, score, heading=heading, vehicle_type=label)
                assigned_tracks.add(t)
                assigned_detections.add(d)

        # Handle unassigned tracks
        for t, tracker in enumerate(self.trackers):
            if t not in assigned_tracks:
                tracker.missed += 1
                tracker.association_history.append(0)
                if len(tracker.association_history) > tracker.confirmation_window:
                    tracker.association_history.pop(0)
                if (
                    tracker.state != "deleted"
                    and tracker.association_history[-tracker.deletion_window :].count(0)
                    >= tracker.deletion_missed_threshold
                ):
                    tracker.state = "deleted"

        # Remove deleted trackers
        self.trackers = [
            tracker for tracker in self.trackers if tracker.state != "deleted"
        ]

        # Create new trackers for unassigned detections
        for d, detection in enumerate(detections):
            if d not in assigned_detections:
                pos = detection["position"]
                dims = detection["dimensions"]
                score = detection["score"]
                heading = detection.get("heading", 0.0)
                label = detection.get("label", 0)

                initial_state = np.array([pos[0], pos[1], pos[2], 0, 0, 0])

                new_tracker = CVKalmanTracker(self.config, self.next_id)
                new_tracker.vehicle_type = label
                new_tracker.confidence = score
                new_tracker.kf.x = initial_state.copy()
                new_tracker.dimensions = dims
                new_tracker.heading = heading

                # Initialize history for the first hit
                new_tracker.hits = 1
                new_tracker.association_history = [1]

                self.next_id += 1
                self.trackers.append(new_tracker)

        return self.get_tracks()

    def compute_cost_matrix(self, detections):
        """Compute cost matrix for data association."""
        if not self.trackers or not detections:
            return np.zeros((len(self.trackers), len(detections)))

        cost_matrix = np.zeros((len(self.trackers), len(detections)))
        for t, tracker in enumerate(self.trackers):
            for d, detection in enumerate(detections):
                # Ensure predicted_state is flat to avoid broadcasting issues with measurement_state
                predicted_state = tracker.kf.x[:3].reshape(-1)
                measurement_state = np.array(detection["position"]).reshape(-1)

                distance_cost = np.linalg.norm(predicted_state - measurement_state)

                # Subclass matching cost
                det_label = detection.get("label", 0)
                if tracker.vehicle_type != det_label:
                    subclass_cost = 10.0
                else:
                    subclass_cost = 0.0

                cost_matrix[t, d] = distance_cost + subclass_cost
        return cost_matrix

    def get_tracks(self):
        """Returns a list of dictionaries representing the current tracks."""
        results = []
        for tracker in self.trackers:
            # Return all tracks that are not deleted.
            # You might want to filter for 'confirmed' only depending on downstream needs.
            # For now, we return everything and let the consumer decide.
            if tracker.state == "deleted":
                continue

            result = {
                "id": tracker.id,
                "label": tracker.vehicle_type,
                "state": tracker.state,
                "position": tracker.kf.x[:3].tolist(),
                "velocity": tracker.kf.x[3:].tolist(),
                "dimensions": tracker.dimensions,
                "heading": tracker.heading,
                "score": tracker.confidence,
            }
            results.append(result)
        return results
