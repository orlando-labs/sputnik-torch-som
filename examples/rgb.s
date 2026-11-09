from io import ByteBuffer
from slowpilot.torch import tensor
from slowpilot.torch_som import SOM

# RGB clustering inspired by the original Torch.rb example.
colors = tensor([
  [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.13, 0.53, 1.0],
  [0.33, 0.4, 0.67], [0.6, 0.5, 1.0], [0.0, 1.0, 0.0],
  [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0],
  [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]
], dtype: "float32")
map = SOM(20, 10).fit!(colors, epochs: 100, seed: 7)
tiles = map.weights.to_a.map |channels|:
  bytes = ByteBuffer(3)
  channels.each |channel|:
    bytes.put!(Math.round(Math.max(0.0, Math.min(1.0, channel)) * 255).to_int)
  "#" + Hex.encode(bytes.bytes)
map.hex_heatmap(tiles, width: 1000, height: 500).save("rgb.png")
