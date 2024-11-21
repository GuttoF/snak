from manim import ORIGIN, Scene  # type: ignore
from manim_ml.neural_network import FeedForwardLayer, NeuralNetwork  # type: ignore


class PN(Scene):
    def construct(self):
        policy_network = NeuralNetwork(
            [
                FeedForwardLayer(6),
                FeedForwardLayer(10),
                FeedForwardLayer(10),
                FeedForwardLayer(10),
                FeedForwardLayer(4),
            ],
            layer_spacing=1,
        )

        # critic_network = NeuralNetwork([
        #  FeedForwardLayer(6),
        #  FeedForwardLayer(10),
        #  FeedForwardLayer(10),
        #  FeedForwardLayer(10),
        #  FeedForwardLayer(1)
        # ], layer_spacing=0.6)

        policy_network.move_to(ORIGIN)
        # critic_network.move_to(ORIGIN)
        self.add(policy_network)

