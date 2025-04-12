# Advanced Traffic Simulation (Python + Pygame)

## Overview

This project is a fun exploration of traffic simulation concepts, built using Python and the Pygame library. It simulates vehicles navigating a road network with intersections, adaptive traffic lights, and dynamic routing. Watch cars try to get around, see how traffic lights adapt, and observe how emergency vehicles get priority!

## Features

* **Graphical User Interface (GUI):** Visualizes the road network, intersections, traffic lights, and moving vehicles using Pygame.
* **Adaptive Traffic Lights:** Intersection lights change based on the number of waiting vehicles, attempting to optimize flow.
* **A\* Routing:** Vehicles use the A\* search algorithm to calculate the fastest route to their destination, considering different road speed limits.
* **Emergency Vehicle (EV) Preemption:** Emergency vehicles request priority at intersections, causing lights to change in their favor.
* **Congestion Visualization:** Roads change color (from gray/cyan towards orange) to indicate traffic density.
* **Varied Road Network:** Includes roads with different speed limits (simulating standard roads and highways).
* **Basic Vehicle Dynamics:** Vehicles accelerate, decelerate, and attempt to maintain safe following distances.

## Requirements

* Python 3.x
* Pygame library: `pip install pygame`

## How to Run

1.  Make sure you have Python 3 and Pygame installed.
2.  Save the simulation code as a Python file (e.g., `traffic_sim.py`).
3.  Open a terminal or command prompt.
4.  Navigate to the directory where you saved the file.
5.  Run the script using:
    ```bash
    python traffic_sim.py
    ```
6.  A Pygame window should open displaying the simulation.

## Controls

* **Spacebar:** Pause / Resume the simulation.
* **'A' Key:** Manually add a few more random vehicles to the simulation (while running).
* **Close Window:** Stop the simulation.

## Potential Enhancements & Fun Ideas

* **More Realistic Physics:** Implement more detailed acceleration/braking models.
* **Driver Behavior:** Add variations in driver behavior (e.g., different desired speeds, reaction times).
* **Real Map Data:** Integrate libraries like `osmnx` to load road networks from OpenStreetMap for real locations.
* **UI Improvements:** Add buttons for controls, display more statistics (average speed, wait times), allow clicking on elements for info.
* **Advanced Traffic Light Algorithms:** Implement more sophisticated adaptive algorithms (e.g., based on predicted arrivals).
* **Accidents/Events:** Simulate random events like accidents or road closures that affect routing.
* **Multi-lane Roads:** Expand roads to have multiple lanes with overtaking logic.

## Note

This was built primarily as a fun project to explore simulation concepts. Feel free to experiment and modify it!

