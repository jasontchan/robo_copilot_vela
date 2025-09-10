from roboenv import Corruptions, SignalCorruptor

XYZ = [0, 1, 2]
RPY = [3, 4, 5]
G = [6]
ALL_CHANNELS = XYZ + RPY + G

corruptor = SignalCorruptor()

# Delay the first 3 channels by up to 5 steps, but only 30% of the time:
corruptor.register(Corruptions["delay"](channels=ALL_CHANNELS, delay_steps=3), prob=0.95)

# # Randomly scale all translational axes between 0.5× and 1.5×, every call:
# corruptor.register(Corruptions["random_scale"](channels=XYZ + RPY, scale_range=(-1.0, 1.0)), prob=0.5)

# # Occasionally zero out a random rotational axis (roll/pitch/yaw):
# corruptor.register(Corruptions["random_zero"](channels=[3, 4, 5]), prob=0.2)

# # Swap x and y signals with 10% chance:
# corruptor.register(Corruptions["random_swap"](channels=[0, 1]), prob=0.9)

# Negate any channel with 50% per-channel chance, but only 5% overall:
corruptor.register(Corruptions["random_negate"](channels=XYZ + RPY, p_channel=0.5), prob=0.5)
