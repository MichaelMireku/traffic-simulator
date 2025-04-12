import time
import random
import math
import pygame
import heapq # For A* priority queue

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 1200, 800 # Screen size
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Smart Traffic Simulation V2") # Updated caption
FONT_SMALL = pygame.font.SysFont(None, 20)
FONT_MEDIUM = pygame.font.SysFont(None, 24)
CLOCK = pygame.time.Clock()

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GRAY = (150, 150, 150)
DARK_GRAY = (100, 100, 100)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255) # Color for faster roads

# --- Simulation Parameters ---
INTERSECTION_COUNT = 10 # Increased number of intersections
DEFAULT_ROAD_SPEED = 15 # m/s (approx 54 km/h) - For standard roads
HIGHWAY_ROAD_SPEED = 25 # m/s (approx 90 km/h) - For faster roads
MAX_SPEED_FACTOR = 1.2 # Vehicles can slightly exceed road speed limit
AVG_VEHICLE_LENGTH = 5
MIN_DISTANCE_BETWEEN_VEHICLES = 3
TIME_STEP = 0.1
# Adaptive Light Parameters
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 25
YELLOW_LIGHT_DURATION = 3
BASE_RED_LIGHT_DURATION = 15
EMERGENCY_PREEMPTION_DISTANCE = 100
# A* Routing Parameters
ASTAR_HEURISTIC_FACTOR = 1.0 # Heuristic based on distance
# GUI Parameters
VEHICLE_WIDTH = 6
VEHICLE_HEIGHT = 10
ROAD_WIDTH = 8

# --- Global Variables ---
intersections = {}
roads = {}
vehicles = []

# --- Helper Functions ---
def get_screen_pos(intersection_id):
    """Maps intersection ID to screen coordinates for the new layout."""
    # Less symmetrical layout
    positions = {
        0: (150, 150), 1: (400, 100), 2: (700, 150), 3: (1050, 100),
        4: (150, 400), 5: (450, 350), 6: (650, 450), 7: (1000, 350),
        8: (300, 650), 9: (750, 600)
        # Add more if INTERSECTION_COUNT increases further
    }
    return positions.get(intersection_id, (random.randint(50, WIDTH-50), random.randint(50, HEIGHT-50))) # Random fallback

def euclidean_distance(pos1, pos2):
    """Calculates straight-line distance between two points."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# --- A* Routing ---
def heuristic(node_id, goal_id):
    """A* heuristic: Euclidean distance / max possible speed (optimistic time)."""
    start_pos = get_screen_pos(node_id)
    end_pos = get_screen_pos(goal_id)
    dist = euclidean_distance(start_pos, end_pos)
    # Use highest possible speed for heuristic to ensure admissibility
    return (dist / HIGHWAY_ROAD_SPEED) * ASTAR_HEURISTIC_FACTOR

def a_star_search(start_node_id, end_node_id):
    """Finds the fastest path using A*, considering road speeds."""
    if start_node_id not in intersections or end_node_id not in intersections:
        return None

    open_set = [(0, start_node_id)] # Priority queue (f_score, node_id)
    came_from = {}

    # g_score now represents the time taken to reach the node
    g_score = {node_id: float('inf') for node_id in intersections}
    g_score[start_node_id] = 0

    # f_score is estimated total time (time so far + heuristic estimate)
    f_score = {node_id: float('inf') for node_id in intersections}
    f_score[start_node_id] = heuristic(start_node_id, end_node_id)

    while open_set:
        current_f_score, current_id = heapq.heappop(open_set)

        # Optimization: If we found a shorter path already, skip
        if current_f_score > f_score[current_id]:
            continue

        if current_id == end_node_id:
            # Reconstruct path
            path = []
            temp = current_id
            while temp in came_from:
                path.append(temp)
                temp = came_from[temp]
            path.append(start_node_id)
            return path[::-1]

        if current_id in intersections:
             # Check if intersection object exists
            if intersections[current_id] is None: continue # Skip if intersection somehow became None

            for road_id in intersections[current_id].outgoing_roads:
                road = roads.get(road_id) # Use get for safety
                if not road: continue # Skip if road doesn't exist

                neighbor_id = road.end_intersection
                # Ensure neighbor exists before accessing scores
                if neighbor_id not in g_score: continue

                # Cost of traversing this edge = estimated time = length / road_speed
                edge_cost = road.length_meters / road.max_speed if road.max_speed > 0 else float('inf')
                tentative_g_score = g_score[current_id] + edge_cost

                if tentative_g_score < g_score[neighbor_id]:
                    # Found a faster path to neighbor
                    came_from[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g_score
                    f_score[neighbor_id] = tentative_g_score + heuristic(neighbor_id, end_node_id)
                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))

    return None # No path found

# --- Classes ---
class Vehicle:
    def __init__(self, id, route, is_emergency=False):
        self.id = id
        self.route = route if route else [] # Ensure route is a list
        self.current_road_id = None
        self.position = 0
        # Start speed relative to the first road's speed limit
        self.speed = 0 # Will be set when added to road
        self.acceleration = random.uniform(0.5, 1.5) # Adjusted acceleration range
        self.length = AVG_VEHICLE_LENGTH
        self.is_emergency = is_emergency
        self.color = PURPLE if is_emergency else BLUE
        self.requesting_preemption = False

        if is_emergency:
           # EV acceleration is higher
           self.acceleration *= 2.0

    def update_speed(self, leading_vehicle=None):
        """Updates speed considering leading vehicle and road speed limit."""
        if not self.current_road_id or self.current_road_id not in roads:
             # print(f"Warning: V{self.id} has invalid current_road_id {self.current_road_id}")
             self.speed = 0
             return # Cannot update speed if not on a valid road

        road = roads[self.current_road_id]
        road_speed_limit = road.max_speed

        # Target speed: slightly above road limit, capped by global max, adjusted for EV
        max_vehicle_speed = road_speed_limit * MAX_SPEED_FACTOR
        # *** Potential Bug Fix: EV speed factor was applied to EMERGENCY_PREEMPTION_DISTANCE instead of a speed factor ***
        # Let's define a separate EV speed factor
        EV_SPEED_BOOST_FACTOR = 1.5
        if self.is_emergency:
             # Apply EV speed boost correctly
             max_vehicle_speed *= EV_SPEED_BOOST_FACTOR

        target_speed = max_vehicle_speed
        follow_speed = target_speed # Speed if no one is ahead

        if leading_vehicle:
            distance = leading_vehicle.position - self.position - self.length
            # Dynamic safe distance: base + reaction time * speed
            safe_distance = MIN_DISTANCE_BETWEEN_VEHICLES + max(0, self.speed * 0.8)

            if distance < safe_distance:
                # Adjust speed based on distance and leading speed
                follow_speed = max(0, leading_vehicle.speed + (distance - safe_distance) * 0.6) # Smoother follow
                if distance < MIN_DISTANCE_BETWEEN_VEHICLES:
                    follow_speed = max(0, self.speed - 6 * TIME_STEP) # Harder brake if too close
            # else: maintain desired speed or accelerate towards it

        # Adjust speed towards the minimum of follow_speed and max_vehicle_speed
        desired_speed = min(follow_speed, max_vehicle_speed)

        if self.speed < desired_speed:
            self.speed += self.acceleration * TIME_STEP
        elif self.speed > desired_speed:
            # Use a braking deceleration (can be different from acceleration)
            braking_deceleration = self.acceleration * 1.5
            self.speed -= braking_deceleration * TIME_STEP

        self.speed = max(0, min(self.speed, max_vehicle_speed * 1.1)) # Allow slight overshoot temp.

    def move(self):
        """Moves the vehicle."""
        self.position += self.speed * TIME_STEP
        if self.current_road_id and self.current_road_id in roads:
            road = roads[self.current_road_id]
            self.position = min(self.position, road.length_meters)

            # --- Emergency Vehicle Preemption Request ---
            if self.is_emergency:
                dist_to_intersection = road.length_meters - self.position
                # Trigger if close AND moving towards intersection
                if dist_to_intersection < EMERGENCY_PREEMPTION_DISTANCE and self.speed > 1:
                    self.requesting_preemption = True
                else:
                    self.requesting_preemption = False
            else:
                self.requesting_preemption = False
        elif self.current_road_id:
             # Road ID exists but road object doesn't - indicates an issue
             # print(f"Warning: V{self.id} move() called with valid road ID {self.current_road_id} but road not found.")
             self.current_road_id = None # Clear invalid road ID


    def draw(self):
        """Draws the vehicle as a rectangle oriented along the road."""
        if not self.current_road_id or self.current_road_id not in roads: return

        road = roads[self.current_road_id]
        start_pos = road.start_pos
        end_pos = road.end_pos

        ratio = self.position / road.length_meters if road.length_meters > 0 else 0
        ratio = max(0, min(1, ratio))

        center_x = start_pos[0] + (end_pos[0] - start_pos[0]) * ratio
        center_y = start_pos[1] + (end_pos[1] - start_pos[1]) * ratio

        angle_rad = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        angle_deg = -math.degrees(angle_rad)

        try:
            # Create a surface for the vehicle rectangle
            vehicle_surf = pygame.Surface((VEHICLE_WIDTH, VEHICLE_HEIGHT), pygame.SRCALPHA)
            vehicle_surf.fill(self.color)

            # Rotate the surface
            rotated_surf = pygame.transform.rotate(vehicle_surf, angle_deg)
            rotated_rect = rotated_surf.get_rect(center=(int(center_x), int(center_y)))

            SCREEN.blit(rotated_surf, rotated_rect)
        except pygame.error as e:
            print(f"Error drawing vehicle {self.id}: {e}")
            print(f"  Position: ({center_x}, {center_y}), Angle: {angle_deg}")
            # Optionally draw a simple circle as fallback
            pygame.draw.circle(SCREEN, self.color, (int(center_x), int(center_y)), 5)


    def __str__(self):
         road_info = f"on R{self.current_road_id}" if self.current_road_id else "off-road"
         return f"V{self.id} (Spd: {self.speed:.1f}, Pos: {self.position:.1f} {road_info}, Em: {self.is_emergency})"


class Intersection:
    def __init__(self, id):
        self.id = id
        # Traffic lights: {road_id: {'state': 'green'/'yellow'/'red', 'timer': float}}
        self.traffic_lights = {}
        self.incoming_roads = [] # List of road IDs
        self.outgoing_roads = [] # List of road IDs
        self.pos = get_screen_pos(id)

        # Adaptive control state
        self.current_green_road = None # Road ID that is currently green/yellow
        self.current_state_timer = 0 # Time remaining in current green/yellow state
        self.preempted_by = None # Road ID that has EV preemption

    def add_incoming_road(self, road_id):
        if road_id not in self.incoming_roads:
            self.incoming_roads.append(road_id)
            # Initialize light state (default to red)
            self.traffic_lights[road_id] = {'state': 'red', 'timer': BASE_RED_LIGHT_DURATION}
            # Start the cycle if this is the first road and it's not already running
            if self.current_green_road is None and self.incoming_roads:
                 self.set_next_green(road_id, MIN_GREEN_TIME)

    def add_outgoing_road(self, road_id):
         if road_id not in self.outgoing_roads:
            self.outgoing_roads.append(road_id)

    def check_for_preemption(self):
        """Checks if an EV requires preemption."""
        requesting_ev_road = None
        min_dist = float('inf')

        for road_id in self.incoming_roads:
            road = roads.get(road_id)
            if not road: continue

            for vehicle in road.vehicles:
                if vehicle.requesting_preemption:
                    dist = road.length_meters - vehicle.position
                    if dist < min_dist:
                        min_dist = dist
                        requesting_ev_road = road_id
                    break # Only one EV per road needs to trigger
        return requesting_ev_road


    def update_lights(self):
        """Adaptive traffic light control with EV preemption."""
        if not self.incoming_roads: return # No roads, nothing to do

        # Ensure cycle is running if it stopped somehow
        if self.current_green_road is None or self.current_green_road not in self.traffic_lights:
             if self.incoming_roads:
                  self.set_next_green(self.incoming_roads[0], MIN_GREEN_TIME)
             else:
                  return # Still no roads

        # --- Emergency Vehicle Preemption ---
        preemption_request_road = self.check_for_preemption()

        if preemption_request_road:
            if self.preempted_by != preemption_request_road:
                # print(f"Intersection {self.id}: EV Preemption requested by R{preemption_request_road}")
                self.preempted_by = preemption_request_road
                for r_id in self.incoming_roads:
                    state_info = self.traffic_lights.get(r_id)
                    if not state_info: continue

                    if r_id == preemption_request_road:
                        state_info['state'] = 'green'
                        state_info['timer'] = MAX_GREEN_TIME # Hold green
                    else:
                        # If currently green/yellow, switch to red quickly
                        if state_info['state'] != 'red':
                           state_info['state'] = 'yellow' # Quick yellow phase
                           state_info['timer'] = 1.0 # Very short yellow
                        # else: # Already red, ensure it stays red
                           # state_info['state'] = 'red' # Already red
                           # state_info['timer'] = MAX_GREEN_TIME # Hold red (redundant if already red?)

                # Ensure current_green_road reflects the preempting road
                self.current_green_road = preemption_request_road
                self.current_state_timer = MAX_GREEN_TIME # Reset timer for preemption duration

            # Keep preempted state active (timer counts down below)
            self.current_state_timer = max(self.current_state_timer, MIN_GREEN_TIME) # Ensure minimum green


        elif self.preempted_by:
            # EV has passed or stopped requesting, end preemption
            # print(f"Intersection {self.id}: Ending preemption by R{self.preempted_by}")
            self.preempted_by = None
            # Force cycle change immediately to recalculate priorities
            self.current_state_timer = 0
            # Set all lights red briefly before starting normal cycle
            for r_id in self.incoming_roads:
                 state_info = self.traffic_lights.get(r_id)
                 if state_info:
                     state_info['state'] = 'red'
                     state_info['timer'] = 1.0 # Short red phase

        # --- Normal Adaptive Cycle (if not preempted) ---
        if not self.preempted_by:
            self.current_state_timer -= TIME_STEP

            # Check if current_green_road is still valid
            if self.current_green_road not in self.traffic_lights:
                 # Invalid state, recover
                 if self.incoming_roads:
                      self.set_next_green(self.incoming_roads[0], MIN_GREEN_TIME)
                 return

            if self.current_state_timer <= 0:
                current_state = self.traffic_lights[self.current_green_road]['state']
                current_green_road_obj = roads.get(self.current_green_road) # Get road object

                if current_state == 'green':
                    # Check conditions to switch
                    green_time_remaining = self.traffic_lights[self.current_green_road]['timer']
                    time_since_green_start = MAX_GREEN_TIME - green_time_remaining # Approx time green has been on

                    waiting_vehicles_current = current_green_road_obj.get_waiting_vehicle_count(approaching=True) if current_green_road_obj else 0

                    max_waiting_other = 0
                    for r_id in self.incoming_roads:
                        if r_id != self.current_green_road:
                             road_obj = roads.get(r_id)
                             if road_obj:
                                  max_waiting_other = max(max_waiting_other, road_obj.get_waiting_vehicle_count())


                    # Condition to switch: Max time reached OR (Min time reached AND (significant demand elsewhere OR current demand low))
                    switch_now = False
                    if green_time_remaining <= 0: # Max time reached (timer counts down)
                        switch_now = True
                    elif time_since_green_start >= MIN_GREEN_TIME: # Min green time passed
                        if max_waiting_other > 3: # Significant demand elsewhere
                             switch_now = True
                        elif waiting_vehicles_current < 1 and max_waiting_other > 0: # Low current demand, some demand elsewhere
                             switch_now = True


                    if switch_now:
                        # Transition to yellow
                        self.traffic_lights[self.current_green_road]['state'] = 'yellow'
                        self.traffic_lights[self.current_green_road]['timer'] = YELLOW_LIGHT_DURATION
                        self.current_state_timer = YELLOW_LIGHT_DURATION
                    else:
                        # Extend green (timer continues counting down)
                         self.current_state_timer = 0.1 # Check again soon
                         self.traffic_lights[self.current_green_road]['timer'] = max(0, green_time_remaining - TIME_STEP)


                elif current_state == 'yellow':
                    # Transition to red, find next green
                    self.traffic_lights[self.current_green_road]['state'] = 'red'
                    self.traffic_lights[self.current_green_road]['timer'] = BASE_RED_LIGHT_DURATION # Placeholder

                    # Find next green based on longest waiting queue
                    best_next_road = None
                    max_waiting = -1

                    # Consider roads in a somewhat cyclic order for fairness
                    try:
                        start_idx = self.incoming_roads.index(self.current_green_road)
                    except ValueError:
                        start_idx = 0 # Fallback if current green road isn't in list anymore

                    possible_next = []
                    for i in range(len(self.incoming_roads)):
                        check_idx = (start_idx + i) % len(self.incoming_roads) # Check current first for waiting count
                        r_id = self.incoming_roads[check_idx]
                        road_obj = roads.get(r_id)
                        wait_count = road_obj.get_waiting_vehicle_count() if road_obj else 0
                        # Only consider roads currently red for becoming next green
                        if self.traffic_lights[r_id]['state'] == 'red':
                             possible_next.append((wait_count, r_id))


                    if not possible_next:
                         # If all roads are somehow green/yellow (shouldn't happen), default to next in cycle
                         next_idx = (start_idx + 1) % len(self.incoming_roads)
                         best_next_road = self.incoming_roads[next_idx]
                         max_waiting = 0
                    else:
                        # Sort by waiting count (descending)
                        possible_next.sort(key=lambda x: x[0], reverse=True)
                        best_next_road = possible_next[0][1] # Road ID with most waiting
                        max_waiting = possible_next[0][0]


                    # Calculate green duration based on waiting cars
                    green_duration = max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, MIN_GREEN_TIME + max_waiting * 1.2))
                    self.set_next_green(best_next_road, green_duration)

        # Decrement individual timers (less critical now, but helps track red duration)
        # for r_id in self.incoming_roads:
        #      state_info = self.traffic_lights.get(r_id)
        #      if state_info:
        #           state_info['timer'] = max(0, state_info['timer'] - TIME_STEP)


    def set_next_green(self, road_id, duration):
        """Sets the specified road to green."""
        if road_id not in self.traffic_lights:
             if not self.incoming_roads: return # No roads to choose from
             road_id = self.incoming_roads[0] # Fallback
             duration = MIN_GREEN_TIME

        self.current_green_road = road_id
        self.traffic_lights[road_id]['state'] = 'green'
        self.traffic_lights[road_id]['timer'] = duration # This timer counts down green time
        self.current_state_timer = duration # This timer tracks state duration

        # Ensure all others are red
        for r_id in self.incoming_roads:
            if r_id != road_id:
                 state_info = self.traffic_lights.get(r_id)
                 if state_info:
                     if state_info['state'] != 'red':
                         # Force red immediately if it was yellow/green
                         state_info['state'] = 'red'
                     # Estimate red timer (mainly for display/info purposes)
                     state_info['timer'] = duration + YELLOW_LIGHT_DURATION


    def get_light_state(self, incoming_road_id):
        return self.traffic_lights.get(incoming_road_id, {'state': 'red'})['state']

    def draw(self):
        """Draws intersection and light states more clearly."""
        pygame.draw.circle(SCREEN, DARK_GRAY, self.pos, 15)
        pygame.draw.circle(SCREEN, BLACK, self.pos, 15, 2)

        for road_id in self.incoming_roads:
            road = roads.get(road_id)
            if not road: continue

            state = self.get_light_state(road_id)
            color = RED
            if state == 'green': color = GREEN
            elif state == 'yellow': color = YELLOW

            # Calculate position for the light indicator
            angle_rad = math.atan2(self.pos[1] - road.start_pos[1], self.pos[0] - road.start_pos[0])
            offset = 25 # Distance from intersection center
            light_x = self.pos[0] - math.cos(angle_rad) * offset
            light_y = self.pos[1] - math.sin(angle_rad) * offset

            pygame.draw.circle(SCREEN, color, (int(light_x), int(light_y)), 7)
            pygame.draw.circle(SCREEN, BLACK, (int(light_x), int(light_y)), 7, 1)


class Road:
    def __init__(self, id, start_intersection_id, end_intersection_id, max_speed=DEFAULT_ROAD_SPEED):
        self.id = id
        self.start_intersection = start_intersection_id
        self.end_intersection = end_intersection_id
        self.vehicles = [] # List of Vehicle objects
        self.start_pos = get_screen_pos(start_intersection_id)
        self.end_pos = get_screen_pos(end_intersection_id)
        self.length_visual = euclidean_distance(self.start_pos, self.end_pos)
        self.length_meters = self.length_visual * 1.0 # Simple scaling: 1 pixel = 1 meter
        self.max_speed = max_speed
        self.congestion_level = 0.0
        self.color = CYAN if max_speed > DEFAULT_ROAD_SPEED else GRAY

    def add_vehicle(self, vehicle):
        if not isinstance(vehicle, Vehicle):
             print(f"Error: Attempted to add non-Vehicle object to Road {self.id}")
             return
        vehicle.current_road_id = self.id
        # Set initial speed relative to this road's limit
        vehicle.speed = random.uniform(self.max_speed * 0.5, self.max_speed * 0.9)
        # *** Bug Fix: Was multiplying speed by distance, should be factor ***
        EV_SPEED_BOOST_FACTOR = 1.5 # Define factor here or globally
        if vehicle.is_emergency:
             vehicle.speed *= EV_SPEED_BOOST_FACTOR
        self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle):
        if not isinstance(vehicle, Vehicle): return # Safety check

        vehicle.current_road_id = None # Clear road association
        # Use try-except for safe removal from lists
        try:
            self.vehicles.remove(vehicle)
        except ValueError:
            # print(f"Warning: Attempted to remove V{vehicle.id} from R{self.id}, but not found in road list.")
            pass
        try:
            vehicles.remove(vehicle) # Remove from global list
        except ValueError:
            # print(f"Warning: Attempted to remove V{vehicle.id} from global list, but not found.")
            pass


    def get_waiting_vehicle_count(self, approaching=False):
        """Counts vehicles stopped/slow near intersection OR approaching green."""
        count = 0
        for v in self.vehicles:
            dist_to_end = self.length_meters - v.position
            if approaching:
                 # Count vehicles within a certain distance, regardless of speed,
                 # to gauge pressure on the green light.
                 if dist_to_end < 60: # Consider vehicles within 60m
                      count += 1
            else:
                # Count vehicles waiting at red/yellow: close AND slow/stopped
                if dist_to_end < 40 and v.speed < 3:
                    count += 1
        return count


    def update_congestion(self):
        """Calculates road congestion level."""
        if self.length_meters <= 0:
             self.congestion_level = 1.0 if self.vehicles else 0.0
             return
        total_vehicle_length = sum(v.length + MIN_DISTANCE_BETWEEN_VEHICLES for v in self.vehicles)
        self.congestion_level = min(1.0, total_vehicle_length / self.length_meters)


    def update_road(self):
        """Moves vehicles, handles lights, transfers vehicles."""
        vehicles_to_remove = []
        vehicles_to_transfer = {} # {vehicle_obj: next_road_id}

        # Sort vehicles by position for easier processing (leading vehicle first)
        self.vehicles.sort(key=lambda v: v.position, reverse=True)
        self.update_congestion()

        end_intersection = intersections.get(self.end_intersection)
        if not end_intersection:
             # print(f"Warning: End intersection {self.end_intersection} not found for Road {self.id}")
             # Remove all vehicles? Or just skip update? Let's skip.
             return

        # --- Update vehicles on this road ---
        for i, vehicle in enumerate(self.vehicles):
            # Ensure vehicle object is valid
            if not isinstance(vehicle, Vehicle):
                 print(f"Warning: Non-vehicle object found in R{self.id} vehicles list.")
                 vehicles_to_remove.append(vehicle) # Mark for removal
                 continue

            leading_vehicle = self.vehicles[i-1] if i > 0 and isinstance(self.vehicles[i-1], Vehicle) else None
            light_state = end_intersection.get_light_state(self.id)

            dist_to_light = self.length_meters - vehicle.position
            effective_stop_dist = vehicle.length / 2 + MIN_DISTANCE_BETWEEN_VEHICLES

            # --- Traffic Light Stop Logic ---
            should_stop_for_light = False
            # Only apply stop logic if vehicle is close enough to potentially need to stop
            # Use a larger distance check based on potential stopping distance
            potential_stop_dist = (vehicle.speed**2) / (2 * (vehicle.acceleration * 1.5)) if vehicle.acceleration > 0 else 50 # Estimate stopping distance
            check_light_distance = max(effective_stop_dist + 5, potential_stop_dist) # Check if within stopping range

            if dist_to_light < check_light_distance:
                if light_state == 'red':
                    if dist_to_light > effective_stop_dist: # Check if actually before the stop line
                         should_stop_for_light = True
                elif light_state == 'yellow' and not vehicle.is_emergency:
                    time_to_cross = (dist_to_light / vehicle.speed) if vehicle.speed > 0.1 else float('inf')
                    # Stop if estimated time to clear intersection > yellow duration
                    if time_to_cross > YELLOW_LIGHT_DURATION * 0.9:
                         should_stop_for_light = True

            # --- Speed Update ---
            if should_stop_for_light:
                target_stop_pos = self.length_meters - effective_stop_dist
                dist_to_stop = target_stop_pos - vehicle.position
                # Apply braking based on distance to target stop position
                brake_decel = max(1.0, (vehicle.speed**2) / (2 * max(1, dist_to_stop)) if dist_to_stop > 0 else 6) # Deceleration needed
                vehicle.speed = max(0, vehicle.speed - brake_decel * TIME_STEP)
                # Prevent overshoot
                if dist_to_stop <= 0.1 and vehicle.speed <= 0.1:
                    vehicle.speed = 0
                    vehicle.position = target_stop_pos
            else:
                # Normal speed update considering leading vehicle
                vehicle.update_speed(leading_vehicle)

            # --- Movement ---
            vehicle.move() # Updates position and EV preemption request flag

            # --- Reaching End of Road ---
            # Check if vehicle is at or past the intended end point
            if vehicle.position >= self.length_meters:
                vehicle.position = self.length_meters # Clamp position

                # Determine next step based on route
                try:
                    # Ensure route is valid list
                    if not isinstance(vehicle.route, list):
                         print(f"Error: V{vehicle.id} has invalid route type: {type(vehicle.route)}. Removing.")
                         vehicles_to_remove.append(vehicle)
                         continue

                    current_route_index = vehicle.route.index(self.end_intersection)
                    if current_route_index + 1 < len(vehicle.route):
                        # Vehicle has more stops in its route
                        next_intersection_id = vehicle.route[current_route_index + 1]

                        # Check if light allows passage
                        if light_state == 'green' or (light_state == 'yellow' and vehicle.is_emergency):
                            # Find the next road segment
                            next_road_found = False
                            for road_id in end_intersection.outgoing_roads:
                                next_road = roads.get(road_id)
                                # Check if this outgoing road leads to the next intersection in the route
                                if next_road and next_road.end_intersection == next_intersection_id:
                                    # Check if there's space on the next road
                                    can_enter = True
                                    for v_other in next_road.vehicles:
                                        # Check if the start of the next road is blocked
                                        if v_other.position < vehicle.length + MIN_DISTANCE_BETWEEN_VEHICLES * 2:
                                            can_enter = False
                                            break
                                    if can_enter:
                                        # Mark vehicle for transfer
                                        vehicles_to_transfer[vehicle] = road_id
                                        next_road_found = True
                                    else:
                                        # Cannot enter next road, stop at intersection end
                                        vehicle.speed = 0
                                        vehicle.position = self.length_meters - 0.1 # Move back slightly
                                    break # Found the correct outgoing road

                            if not next_road_found and vehicle not in vehicles_to_transfer:
                                # If the correct outgoing road wasn't found (routing error)
                                # print(f"Routing Error: No physical road from I{self.end_intersection} to I{next_intersection_id} for V{vehicle.id}. Route: {vehicle.route}. Removing.")
                                vehicles_to_remove.append(vehicle)
                        else:
                             # Stopped at red/yellow light right at the end line
                             vehicle.speed = 0
                             vehicle.position = self.length_meters - 0.1 # Ensure it stays before the line

                    else:
                        # Reached final destination in route
                        vehicles_to_remove.append(vehicle)

                except ValueError:
                     # Current intersection not found in the route list - major routing error
                     # print(f"Critical Routing Error: I{self.end_intersection} not in route for V{vehicle.id}. Route: {vehicle.route}. Removing.")
                     vehicles_to_remove.append(vehicle)
                except IndexError:
                     # Error accessing route index - corrupted route?
                     # print(f"Critical Routing Error: Invalid route index for V{vehicle.id}. Route: {vehicle.route}. Removing.")
                     vehicles_to_remove.append(vehicle)
                except Exception as e:
                     # Catch any other unexpected errors during route processing
                     print(f"Unexpected Error processing route for V{vehicle.id} at I{self.end_intersection}: {e}. Route: {vehicle.route}. Removing.")
                     vehicles_to_remove.append(vehicle)


        # --- Process Transfers and Removals (outside the loop) ---
        for vehicle_obj, next_road_id in vehicles_to_transfer.items():
             # Double check vehicle is still in this road's list before removing
             if vehicle_obj in self.vehicles:
                 self.remove_vehicle(vehicle_obj) # Removes from self.vehicles and global vehicles list
                 vehicle_obj.position = 0 # Reset position for the new road
                 next_road = roads.get(next_road_id)
                 if next_road:
                     next_road.add_vehicle(vehicle_obj) # Adds to next_road.vehicles
                     vehicles.append(vehicle_obj) # IMPORTANT: Add back to global list
                 else:
                      print(f"Error: Failed to transfer V{vehicle_obj.id} to non-existent road R{next_road_id}.")


        # Process removals ensuring we don't try to remove already transferred vehicles
        final_vehicles_to_remove = [v for v in vehicles_to_remove if v not in vehicles_to_transfer]
        for vehicle_obj in final_vehicles_to_remove:
             if vehicle_obj in self.vehicles: # Check if still present
                 self.remove_vehicle(vehicle_obj)


    def draw(self):
        """Draws the road, color-coded for speed limit and congestion."""
        base_color = self.color
        congestion_color = (
            int(base_color[0] + (ORANGE[0] - base_color[0]) * self.congestion_level),
            int(base_color[1] + (ORANGE[1] - base_color[1]) * self.congestion_level),
            int(base_color[2] + (ORANGE[2] - base_color[2]) * self.congestion_level)
        )
        try:
            pygame.draw.line(SCREEN, congestion_color, self.start_pos, self.end_pos, ROAD_WIDTH)
            # Corrected line: Removed the 6th argument '1'
            pygame.draw.line(SCREEN, DARK_GRAY, self.start_pos, self.end_pos, ROAD_WIDTH + 2) # Draw outline
        except TypeError as e:
             print(f"Error drawing road {self.id}: {e}")
             print(f"  Start: {self.start_pos}, End: {self.end_pos}, Width: {ROAD_WIDTH}")
        except Exception as e:
             print(f"Unexpected error drawing road {self.id}: {e}")


    def __str__(self):
        return f"Road {self.id} ({self.start_intersection}->{self.end_intersection}) MaxSpd: {self.max_speed} Veh: {len(self.vehicles)} Cong: {self.congestion_level:.2f}"


# --- Simulation Setup Functions ---

def create_network(intersection_count):
    """Creates the more complex road network with varied speeds."""
    global intersections, roads
    # Clear previous network data
    intersections.clear()
    roads.clear()

    intersections = {i: Intersection(i) for i in range(intersection_count)}
    road_id_counter = 0

    # Define connections: (start_id, end_id, speed_type) - speed_type 'h' for highway
    connections = [
        (0, 1, 's'), (1, 0, 's'), (1, 2, 's'), (2, 1, 's'), (2, 3, 'h'), (3, 2, 'h'), # Top row-ish
        (0, 4, 's'), (4, 0, 's'), (1, 5, 's'), (5, 1, 's'), (2, 6, 's'), (6, 2, 's'), (3, 7, 's'), (7, 3, 's'), # Verticals
        (4, 5, 'h'), (5, 4, 'h'), (5, 6, 's'), (6, 5, 's'), (6, 7, 'h'), (7, 6, 'h'), # Middle row-ish
        (4, 8, 's'), (8, 4, 's'), (5, 8, 's'), (8, 5, 's'), (6, 9, 's'), (9, 6, 's'), (7, 9, 's'), (9, 7, 's'), # Lower connections
        (8, 9, 'h'), (9, 8, 'h'), # Bottom connection
        # Add a few more cross links
        (1, 4, 's'), (4, 1, 's'), (2, 5, 's'), (5, 2, 's'), (6, 3, 's'), (3, 6, 's')
    ]

    added_roads = set() # Track (start, end) tuples to avoid duplicates

    for start_id, end_id, speed_type in connections:
        # Basic validation
        if start_id >= intersection_count or end_id >= intersection_count or start_id < 0 or end_id < 0:
            print(f"Warning: Invalid node ID in connection ({start_id}, {end_id}). Skipping.")
            continue
        if start_id == end_id:
             print(f"Warning: Road connects node {start_id} to itself. Skipping.")
             continue
        # Check for duplicate road definition
        if (start_id, end_id) in added_roads:
             print(f"Warning: Duplicate road definition ({start_id} -> {end_id}). Skipping.")
             continue

        road_speed = HIGHWAY_ROAD_SPEED if speed_type == 'h' else DEFAULT_ROAD_SPEED
        road_id = f"r{road_id_counter}"

        # Create and store the road
        roads[road_id] = Road(road_id, start_id, end_id, max_speed=road_speed)

        # Add outgoing/incoming references to intersections
        if start_id in intersections:
            intersections[start_id].add_outgoing_road(road_id)
        else: print(f"Error: Start intersection {start_id} not found for road {road_id}")

        if end_id in intersections:
            intersections[end_id].add_incoming_road(road_id)
        else: print(f"Error: End intersection {end_id} not found for road {road_id}")

        added_roads.add((start_id, end_id)) # Mark as added
        road_id_counter += 1

    print(f"Network created with {len(intersections)} intersections and {len(roads)} roads.")
    # Sanity check intersections
    for i_id, i_obj in intersections.items():
         if not i_obj.incoming_roads and any(c[1] == i_id for c in connections):
              print(f"Warning: Intersection {i_id} has no incoming roads despite connections.")
         if not i_obj.outgoing_roads and any(c[0] == i_id for c in connections):
              print(f"Warning: Intersection {i_id} has no outgoing roads despite connections.")


def add_vehicles_to_sim(num_vehicles, num_emergency=0):
    """Adds vehicles using A* routing."""
    global vehicles # Ensure we modify the global list
    added_count = 0
    emergency_added = 0
    attempts = 0

    available_nodes = list(intersections.keys())
    if len(available_nodes) < 2:
        print("Not enough intersections to create routes.")
        return

    # Use a temporary list to avoid modifying global list while iterating
    current_global_vehicles = list(vehicles)
    # Generate unique IDs based on current time and random number
    base_id = int(time.time() * 10)

    while (added_count < num_vehicles or emergency_added < num_emergency) and attempts < (num_vehicles + num_emergency) * 5:
        attempts += 1
        is_emergency = emergency_added < num_emergency

        start_node = random.choice(available_nodes)
        end_node = random.choice(available_nodes)
        while end_node == start_node:
            end_node = random.choice(available_nodes)

        route = a_star_search(start_node, end_node)

        if route and len(route) >= 2:
            # Find the first road segment of the route
            first_road_id = None
            start_intersection = intersections.get(route[0])
            if start_intersection:
                for r_id in start_intersection.outgoing_roads:
                    road = roads.get(r_id)
                    # Check if the road object exists and its end matches the route's next step
                    if road and road.end_intersection == route[1]:
                        first_road_id = r_id
                        break

            if first_road_id:
                start_road = roads[first_road_id]
                # Ensure start_road exists
                if not start_road:
                     # print(f"Error: First road {first_road_id} for route {route} not found.")
                     continue

                # Generate unique ID
                new_id = base_id + len(current_global_vehicles) + added_count + emergency_added

                vehicle = Vehicle(new_id, route, is_emergency=is_emergency)

                # Check space at the start of the road
                can_add = True
                for v in start_road.vehicles:
                    # Check position relative to new vehicle length
                    if v.position < vehicle.length + MIN_DISTANCE_BETWEEN_VEHICLES:
                        can_add = False
                        break

                if can_add:
                    start_road.add_vehicle(vehicle) # Adds to road's list
                    vehicles.append(vehicle) # Add to global list *explicitly*
                    if is_emergency:
                        emergency_added += 1
                    else:
                        added_count += 1
                # else: Cannot add, start blocked
            # else: No road found for first segment
        # else: No route found by A*

    # Final check if all requested vehicles were added
    # if added_count < num_vehicles or emergency_added < num_emergency:
    #      print(f"Warning: Added {added_count}/{num_vehicles} regular, {emergency_added}/{num_emergency} emergency vehicles after {attempts} attempts.")


# --- Main Simulation Loop ---
def run_simulation():
    global vehicles # Make sure updates affect the global list
    running = True
    simulation_time = 0 # Step counter
    last_vehicle_add_time = -60 # Sim seconds
    paused = False

    # Initialize network and vehicles
    create_network(INTERSECTION_COUNT)
    vehicles.clear() # Clear any vehicles from previous runs if applicable
    add_vehicles_to_sim(num_vehicles=35, num_emergency=3)

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Simulation Paused" if paused else "Simulation Resumed")
                if event.key == pygame.K_a:
                     if not paused: # Only add if not paused
                         add_vehicles_to_sim(num_vehicles=5, num_emergency=random.choice([0,1]))
                         print(f"Added vehicles. Total: {len(vehicles)}")
                     else:
                         print("Cannot add vehicles while paused.")

        # --- Simulation Update Logic ---
        if not paused:
            current_sim_time_sec = simulation_time * TIME_STEP

            # Add new vehicles periodically
            if current_sim_time_sec - last_vehicle_add_time >= 10: # Add every 10 sim seconds
                add_vehicles_to_sim(num_vehicles=random.randint(3, 6), num_emergency=random.choice([0,0,1]))
                last_vehicle_add_time = current_sim_time_sec

            # --- Update Game State ---
            # 1. Update Intersections (Traffic Lights)
            for i in intersections.values():
                i.update_lights()

            # 2. Update Roads (Vehicle Movement, Transfers, Congestion)
            # Iterate over a copy of road values in case roads dict changes (unlikely here)
            for r in list(roads.values()):
                 r.update_road() # This handles vehicle movement, speed, transfers, removals

            simulation_time += 1 # Increment step counter

        # --- Drawing ---
        SCREEN.fill(WHITE) # Clear screen

        # Draw elements: Roads -> Intersections -> Vehicles
        try:
            # Draw roads
            for r in roads.values():
                r.draw()

            # Draw intersections (lights) on top of roads
            for i in intersections.values():
                i.draw()

            # Draw vehicles on top
            # Iterate over a copy of the global list in case it's modified during drawing (e.g., by removal)
            for v in list(vehicles):
                v.draw()
        except Exception as e:
            print(f"Error during drawing phase: {e}")
            # Potentially add more robust error handling or fallback drawing


        # --- Display Info ---
        sim_seconds = simulation_time * TIME_STEP
        time_text = FONT_MEDIUM.render(f"Sim Time: {sim_seconds:.1f}s", True, BLACK)
        # Ensure vehicles list is accessed safely for count
        vehicle_count = len(vehicles) if vehicles is not None else 0
        vehicle_count_text = FONT_MEDIUM.render(f"Vehicles: {vehicle_count}", True, BLACK)
        pause_text = FONT_MEDIUM.render("PAUSED (Space)", True, RED) if paused else None
        add_vehicle_text = FONT_SMALL.render("Add Vehicles (A)", True, BLACK)

        SCREEN.blit(time_text, (10, 10))
        SCREEN.blit(vehicle_count_text, (10, 35))
        SCREEN.blit(add_vehicle_text, (10, 60))
        if pause_text:
            SCREEN.blit(pause_text, (WIDTH // 2 - pause_text.get_width() // 2, 10))


        # --- Update Display ---
        pygame.display.flip()

        # --- Timing ---
        CLOCK.tick(60) # Limit frame rate

    # --- Cleanup ---
    pygame.quit()
    print("Simulation ended.")

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as main_exception:
        print(f"\n--- UNHANDLED EXCEPTION CAUGHT ---")
        import traceback
        traceback.print_exc()
        print(f"--- END OF EXCEPTION ---")
        pygame.quit() # Ensure pygame quits even on error

