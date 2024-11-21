from manim import ORIGIN, Scene  # type: ignore
from manim_ml.neural_network import FeedForwardLayer, NeuralNetwork  # type: ignore


class ADV(Scene):
    def construct(self):
        critic_network = NeuralNetwork(
            [
                FeedForwardLayer(6),
                FeedForwardLayer(10),
                FeedForwardLayer(10),
                FeedForwardLayer(10),
                FeedForwardLayer(1),
            ],
            layer_spacing=1,
        )

        critic_network.move_to(ORIGIN)
        self.add(critic_network)
