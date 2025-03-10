import torch
import re
import gymnasium as gym
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from utils import step, load_dataset, decode, LOCS, set_up_logger
from prompts import base_prompt


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "GRPO-taxi"
DEFAULT_NEGATIVE_REWARD = 0.0

SYSTEM_PROMPT = (
    "You are an intelligent assistant solving the Taxi game. The goal is to pick up the passenger, "
    "navigate to their destination, and drop off the passenger. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user "
    "with the answer which action to take. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


class TaxiGRPOTrainer:
    """Main class to handle the GRPO training for the Taxi environment."""
    
    def __init__(self):
        self.logger = set_up_logger()
        self.env = gym.make("Taxi-v3", render_mode="ansi")
    
    def make_conversation(self, sample):
        """Create a conversation from a sample."""
        user_input = base_prompt.format(state_txt=sample["state_txt"])
        
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            "state_id": sample["state"]
        }
    
    def action_reward(self, completions, **kwargs):
        """Calculate rewards based on actions taken in the environment."""
        actions = [completion[0]["content"] for completion in completions]
        state_txts = [state_txt[1]["content"] for state_txt in kwargs["prompts"]]
        states = kwargs["state"]
        
        rewards = []
        for idx, action in enumerate(actions):
            state_desc = state_txts[idx].split("### Current State:")[-1].split(" ### Output:")[0].strip()
            state = states[idx]
            
            self.logger.debug(f"Completion number {idx}")
            self.logger.debug(f"State number {state}")
            self.logger.debug(f"Action: {action}")
            self.logger.debug(f"State description: {state_desc}")
            
            # Decode current state
            taxi_row, taxi_col, pass_idx_current, dest_idx = decode(state)
            self.logger.debug(f"State positions: {taxi_row, taxi_col, pass_idx_current, dest_idx}")
            
            # Parse the action and check format
            match = re.search(r"<answer>(.*?)</answer>", action)
            if not match:
                rewards.append(DEFAULT_NEGATIVE_REWARD)
                continue
            
            predicted_action = match.group(1)
            self.logger.debug(f"Predicted action: {predicted_action}")
            
            # Check if predicted action is a valid integer
            if not predicted_action.isdigit():
                rewards.append(DEFAULT_NEGATIVE_REWARD)
                continue
            
            predicted_action = int(predicted_action)
            if predicted_action not in range(6):
                rewards.append(DEFAULT_NEGATIVE_REWARD)
                continue
            
            # Check if passenger is already picked up
            picked_up = "Taxi (with passenger)" in state_desc
            self.logger.debug(f"Passenger in taxi: {picked_up}")
            
            # Get passenger location
            pass_loc = LOCS[pass_idx_current] if pass_idx_current < 4 else (taxi_row, taxi_col)
            positions = (taxi_row, taxi_col), pass_loc, LOCS[dest_idx]
            
            # Take action in environment
            next_state, env_reward, done, _, _ = step(self.env, state, predicted_action)
            self.logger.debug(f"Next state: {next_state}")
            
            # Decode next state
            next_state_positions = self._get_positions_from_state(next_state)
            
            # Calculate reward
            action_reward = self._calculate_direction_reward(
                predicted_action, positions, next_state_positions, pass_idx_current, picked_up
            )
            
            total_reward = action_reward
            
            # Override reward for successful drop-off
            if env_reward == 20:
                total_reward = 1
                self.logger.debug("DROP-OFF SUCCESSFUL! Win!")
            
            rewards.append(total_reward)
            self.logger.debug(f"Action reward: {action_reward}, Env reward: {env_reward}")
        
        self.logger.debug(f"REWARDS: {rewards}")
        return rewards
    
    def _get_positions_from_state(self, state):
        """Extract positions from state."""
        taxi_row, taxi_col, pass_idx, dest_idx = decode(state)
        taxi_loc = (taxi_row, taxi_col)
        
        if pass_idx == 4:  # Passenger in taxi
            return taxi_loc, taxi_loc, LOCS[dest_idx]
        else:
            return taxi_loc, LOCS[pass_idx], LOCS[dest_idx]
    
    def _calculate_direction_reward(self, action, positions, next_positions, pass_idx, picked_up):
        """Calculate reward based on movement direction relative to target. """
        self.logger.debug(f"Positions: {positions}")
        
        if not picked_up:
            return self._calculate_pickup_reward(action, positions, next_positions)
        else:
            return self._calculate_dropoff_reward(positions, next_positions)
    
    def _calculate_pickup_reward(self, action, positions, next_positions):
        """Calculate reward for moving toward passenger."""
        self.logger.debug("Passenger not picked up yet. Get distance of taxi and passenger")
        
        # Current positions
        taxi_pos = positions[0]
        pass_pos = positions[1]
        
        # Next positions
        taxi_next_pos = next_positions[0]
        
        # Calculate deltas
        delta_y_current = taxi_pos[0] - pass_pos[0]
        delta_x_current = taxi_pos[1] - pass_pos[1]
        delta_y_next = taxi_next_pos[0] - pass_pos[0]
        delta_x_next = taxi_next_pos[1] - pass_pos[1]
        
        self.logger.debug(f"Delta current: ({delta_y_current}, {delta_x_current})")
        self.logger.debug(f"Delta next: ({delta_y_next}, {delta_x_next})")
        
        # Successful pickup
        if positions[0] == positions[1] and action == 4:
            self.logger.debug("Pick-Up Successful!!")
            return 1.0
        
        # Calculate direction improvements
        delta_y = delta_y_next - delta_y_current
        delta_x = delta_x_next - delta_x_current
        
        # Adjust for special cases
        if pass_pos[0] == 0:  # Passenger at top
            delta_y *= -1
        if pass_pos[1] == 0:  # Passenger at left
            delta_x *= -1
        
        # Special case for lower right corner
        if pass_pos[1] == 3 and taxi_pos[1] >= 3 and taxi_next_pos[1] >= 3:
            delta_x *= -1
        
        # Reward if moving in right direction
        return 1.0 if (delta_y > 0 or delta_x > 0) else DEFAULT_NEGATIVE_REWARD
    
    def _calculate_dropoff_reward(self, positions, next_positions):
        """Calculate reward for moving toward destination."""
        self.logger.debug("Passenger picked up. Get distance of taxi and destination.")
        
        # Current positions
        taxi_pos = positions[0]
        dest_pos = positions[2]
        
        # Next positions
        taxi_next_pos = next_positions[0]
        
        # Calculate deltas
        delta_y_current = taxi_pos[0] - dest_pos[0]
        delta_x_current = taxi_pos[1] - dest_pos[1]
        delta_y_next = taxi_next_pos[0] - dest_pos[0]
        delta_x_next = taxi_next_pos[1] - dest_pos[1]
        
        self.logger.debug(f"Delta current: ({delta_y_current}, {delta_x_current})")
        self.logger.debug(f"Delta next: ({delta_y_next}, {delta_x_next})")
        
        # Calculate direction improvements
        delta_y = delta_y_next - delta_y_current
        delta_x = delta_x_next - delta_x_current
        
        # Adjust for special cases
        if dest_pos[0] == 0:  # Destination at top
            delta_y *= -1
        if dest_pos[1] == 0:  # Destination at left
            delta_x *= -1
        
        # Special case for lower right corner
        if dest_pos[1] == 3 and taxi_pos[1] >= 3 and taxi_next_pos[1] >= 3:
            delta_x *= -1
        
        # Reward if moving in right direction
        return 1.0 if (delta_y > 0 or delta_x > 0) else DEFAULT_NEGATIVE_REWARD
    
    def int_type_reward(self, completions, **kwargs):
        """Reward if predicted action is a valid integer between 0 and 5."""
        rewards = []
        
        for completion in completions:
            content = completion[0]["content"]
            match = re.search(r"<answer>(.*?)</answer>", content)
            
            if not match:
                rewards.append(DEFAULT_NEGATIVE_REWARD)
                continue
            
            predicted_action = match.group(1)
            
            if not predicted_action.isdigit():
                rewards.append(DEFAULT_NEGATIVE_REWARD)
            elif int(predicted_action) not in range(6):
                 # is it a valid integer in range 0..5?
                rewards.append(DEFAULT_NEGATIVE_REWARD)
            else:
                rewards.append(1.0)
        
        self.logger.debug(f"INT REWARDS: {rewards}")
        return rewards
    
    def format_reward(self, completions, **kwargs):
        """Reward if completion follows the required format."""
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]
        rewards = [1.0 if match else DEFAULT_NEGATIVE_REWARD for match in matches]
        
        self.logger.debug(f"FORMAT REWARDS: {rewards}")
        return rewards
    
    def load_data(self):
        """Load and prepare training data."""
        train_dataset, test_dataset = load_dataset()
        
        train_dataset = train_dataset.map(self.make_conversation)
        test_dataset = test_dataset.map(self.make_conversation)
        
        return train_dataset, test_dataset
    
    def create_model(self):
        """Create and configure the model."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            # quantization_config=bnb_config, 
            torch_dtype="auto",
            device_map="auto",
        )
        
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=None,  # ["_proj", "v_proj"]
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def create_training_args(self):
        """Create training arguments."""
        return GRPOConfig(
            output_dir=OUTPUT_DIR,
            learning_rate=1e-5,
            beta=0.01,
            reward_weights=[0.25, 0.25, 0.5],
            per_device_train_batch_size=6,
            remove_unused_columns=False,
            gradient_accumulation_steps=2,
            num_train_epochs=100,
            bf16=True,
            max_completion_length=128,
            num_generations=6,
            max_prompt_length=None,
            report_to=["tensorboard"],
            logging_steps=10,
            push_to_hub=False,
            save_strategy="steps",
            save_steps=10,
            log_completions=True,
            resume_from_checkpoint=None,
        )
    
    def train(self):
        """Run the training process."""
        # Load data
        train_dataset, test_dataset = self.load_data()
        
        # Create model
        model = self.create_model()
        
        # Configure training
        training_args = self.create_training_args()
        
        # Create trainer
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[
                self.format_reward,
                self.int_type_reward,
                self.action_reward
            ],
            args=training_args,
            train_dataset=train_dataset
        )

        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model(training_args.output_dir)


def main():
    """Main entry point."""
    trainer = TaxiGRPOTrainer()
    trainer.train()


if __name__ == "__main__":
    main()