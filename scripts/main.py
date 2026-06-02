from utils import connect_to_coppeliasim, record_episode, randomize_cube_position
import time

def main():
    sim = connect_to_coppeliasim()
    cube_handle = sim.getObject('/Cube')

    for i in range(100):
        randomize_cube_position(sim, cube_handle)  # Randomize cube position at the start of each episode
        print(f"Episode {i+1}: cube randomized.")

        sim.clearInt32Signal('pick_and_place_done') # Clear the signal at the start of each episode

        sim.startSimulation()
        print(f"Episode {i+1}: simulation started.")

        # Polls sensors AND waits for completion — all on this thread.
        # Returns only when pick_and_place_done == 1 or sim stops.
        # Saves the HDF5 file before returning.
        record_episode(sim, i + 1)

        sim.stopSimulation()
        print(f"Episode {i + 1}: Simulation stopped!")

        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)  # Wait until the simulation is fully stopped

if __name__ == "__main__":
    main()