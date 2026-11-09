# torch-som

Kohonen self-organizing maps for Sputnik. Training and assignment use `torch`
tensors. `plot` renders the staggered hexagonal grid as a polygon heatmap.

```sh
sputnik pkg install slowpilot/torch-som --version 0.1.0 --allow-native
```

```sputnik
from slowpilot.torch import tensor
from slowpilot.torch_som import SOM, cosine_similarity, plasma

embeddings = tensor([[1.0, 0.0], [1.0, 0.2], [-1.0, 0.0]], dtype: "float32")
map = SOM(4, 3, metric: :cosine).fit!(embeddings, seed: 7)
assignments = map.predict(embeddings)
similarities = cosine_similarity(embeddings)
map.hex_heatmap(map.neighbor_similarity.to_a, palette: plasma(), label: "Cosine")
  .save("cosine.svg")
```

- `fit!` accepts a nonempty dense `float32` or `float64` tensor with shape
  `[samples, features]`. Batch training initializes from sample rows and uses a
  Gaussian neighborhood over hex grid hops, with its radius decaying to 0.5.
- `metric: :euclidean` (default) selects the nearest prototype by Euclidean
  distance. `metric: :cosine` selects by cosine similarity and learns unit
  directions. A zero vector has similarity zero to every vector, including itself.
- `predict` returns zero-based row-major indices. Ties choose the first
  prototype. `weights` and `weights_2d` return copies in shapes
  `[width * height, features]` and `[height, width, features]`.
- `neighbor_similarity` returns each prototype's mean cosine similarity to
  its immediate hex neighbors. A single-cell map returns zero.
- `cosine_similarity` returns the full pairwise matrix in the supplied order.
  `hex_heatmap` accepts one numeric value or CSS hex color per cell. It returns
  a `plot.Figure` that can be rendered as SVG or PNG.

The RGB example adapts the [original Ruby example](https://github.com/orlando-labs/torch-som/blob/master/example.rb).
From the checkout, run `sputnik run examples/rgb.s --grant ffi --grant fs.write=rgb.png`
to render the learned RGB prototypes.

The Plasma palette samples [Matplotlib's Plasma colormap](https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_cm_listed.py).
