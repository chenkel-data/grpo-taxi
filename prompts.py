base_prompt = """

    Analyze the current game state and generate an action for the next step in form of an integer to move down, up, left, right, pick-up, drop-off. 
    Your objective is to move the taxi to the passenger's location, pick up the passenger, and drive to the destination while avoiding obstacles and drop him off. \n    

    ### Rules \n
    - There are four designated pick-up and drop-off locations in the 5x5 grid world. 
    - The taxi starts off at a random square and the passenger at one of the designated locations.
    - You receive positive rewards if your action brings you closer to the target—either the passenger or the destination—depending on whether the passenger is already inside the taxi.
    - You receive positive rewards for successfully picking up or dropping off the passenger at the correct location. 
    - You receive negative rewards for incorrect attempts to pick-up/drop-off passenger.
    - Once the passenger is dropped off, the episode ends.


    ### Strategy Guidelines \n
    - As the actions for pick up and drop off are special actions and only required excatly once each, avoid to frequently applying these 2 special actions. 
    - Only apply the actions for pick up and drop off only when possible and necessary i.e. the taxi is on the same location as the passenger or at the destination location.
    

    ### Map:\n

    The game world consists of a 5x5 grid enclosed by a border of + and - characters. The taxi, passenger, and destinations are placed in specific locations, and walls (|) restrict movement. 
    The colons (:) signal that the taxi can cross-over. Walls cannot be crossed-over.
    

    ### Actions:\n
    0: Move south
    1: Move north
    2: Move east
    3: Move west
    4: Pick up passenger
    5: Drop off passenger

    ### Current State: \n
    {state_txt}

    ### Output: \n
    - Analyze the game state from the provided state and determine the optimal move.
    - Provide your response in a format as follows:\n\n

    <think><reasoning steps explaining why the action was chosen to reach the goal. Keep the reasoning precise and short.></think>
    <answer>integer</answer>

    Ensure that:\n
    - The 'answer' field contains solely a single integer for one of the 6 valid actions in range 0 to 5 and not any text or multiple integers.\n
    - The 'think' field provides a brief explanation (few words) of why the action is the best choice.\n


    """

map_example_prompt = """

    Analyze the current game state and generate an action for the next step in form of an integer to move down, up, left, right, pick-up, drop-off. 
    Your objective is to move the taxi to the passenger's location, pick up the passenger, and drive to the destination while avoiding obstacles and drop him off. \n    

    ### Rules \n
    The taxi navigates a 5x5 grid with four designated pickup/dropoff locations, starting from a random position while the passenger begins at one of these locations. You earn positive rewards for moving closer to your target (passenger when empty, destination when carrying) and for successful pick ups and drop offs. Incorrect pickup/dropoff attempts result in negative rewards. The episode concludes once the passenger is successfully delivered to the destination.

    ### Strategy Guidelines \n  
    Execute pickup and dropoff actions only when the taxi is at the same location as the passenger or destination, and determine your target based on whether the taxi is empty (target is passenger) or carrying the passenger (target is destination), then select the optimal direction to reach that target while navigating around any walls "|".

    ### Map:\n

    The game world consists of a 5x5 grid enclosed by a border of + and - characters. The taxi, passenger, and destinations are placed in specific locations, and walls (|) restrict movement. 
    The colons (:) signal that the taxi can cross-over. Walls cannot be crossed-over.
    
    ### Example Map:\n
    +---------+, | :Taxi (empty)| : :Passenger waiting|, | : | : : |, | : : : : |, | | : | : |, |destination| : | : |, +---------+
    
    #### Explanation:\n
    - `+---------+` represents the **top and bottom walls** of the grid.  
    - Each new row in the grid **starts with a comma (`,`)**.  
    - In this example:  
        - The **taxi** and the **passenger** are in the first row (**top row**).  
        - The **destination** is in the **bottom row**.  
        - The **taxi is empty**. 
        - The **taxi** cannot move east (right) as there is a wall "|". But the taxi can move left (west) due to the colon ":" or it can move south (down).
    - Furthermore: 
        - "Passenger waiting:Taxi (empty):" means that the taxi is to the right to the passenger and the taxi should move west (left)
        - "|Taxi (empty) Passenger waiting:" means the empty taxi and the passenger are at the same location and the taxis should pick-up the passenger.
        
    ### Actions:\n
    0: Move south
    1: Move north
    2: Move east
    3: Move west
    4: Pick up passenger
    5: Drop off passenger

    ### Current State: \n
    {state_txt}

    ### Output: \n
    - Analyze the game state from the provided state and determine the optimal move.
    - Provide your response in a format as follows:\n\n

    <think><reasoning steps explaining why the action was chosen to reach the goal. Keep the reasoning precise and short.></think>
    <answer>integer</answer>

    Ensure that:\n
    - The 'answer' field contains solely a single integer for one of the 6 valid actions in range 0 to 5 and not any text or multiple integers.\n
    - The 'think' field provides a brief explanation (few words) of why the action is the best choice.\n

    """
