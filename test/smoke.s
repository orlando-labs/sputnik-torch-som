from slowpilot.torch import tensor
from slowpilot.torch_som import SOM, cosine_similarity, plasma

def check(condition, message):
  raise ValueError(message) unless condition

samples = tensor([[2.0, 0.0], [0.0, 3.0], [-4.0, 0.0], [0.0, 0.0]], dtype: "float32")
matrix = cosine_similarity(samples).to_a
check(matrix == [[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0],
  [-1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]], "pairwise cosine including zero rows")
check(cosine_similarity(tensor([[1e30, 0.0], [1e-30, 0.0]], dtype: "float32")).to_a
  == [[1.0, 1.0], [1.0, 1.0]], "cosine remains scale invariant")

map = SOM(3, 2, metric: :cosine).fit!(samples, epochs: 15, seed: 7)
check(map.weights_2d.shape == [2, 3, 2], "prototype grid shape")
check(map.predict(samples).to_a == SOM(3, 2, metric: :cosine)
  .fit!(samples, epochs: 15, seed: 7).predict(samples).to_a, "reproducible initialization")
scaled = tensor([[0.25, 0.0], [-0.75, 0.0]], dtype: "float32")
assignments = map.predict(scaled).to_a
check(assignments[0] != assignments[1], "opposite directions map to different cells")
check(map.predict(scaled * 10).to_a == assignments, "cosine assignments ignore magnitude")
check(map.neighbor_similarity.shape == [6], "one similarity per cell")

single = SOM(1, 1).fit!(samples)
check(single.weights.to_a == [[-0.5, 0.75]], "Euclidean batch mean")
check(single.neighbor_similarity.to_a == [0.0], "single cell has no neighbors")

figure = map.hex_heatmap([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0], palette: plasma())
hexagons = figure.layers[0].data.all? |tile|: tile.outer.count == 6
check(hexagons, "each tile is a hexagon")
svg = figure.render.text
check(svg.contains?("#0d0887") and svg.contains?("#f0f921"), "Plasma endpoints in SVG")
check(figure.render(format: :png).bytes.slice(0, 8).hex == "89504e470d0a1a0a", "PNG signature")
print("torch-som smoke checks passed")
