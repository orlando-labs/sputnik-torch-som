package slowpilot.torch_som

import slowpilot.torch as torch
import slowpilot.plot as plot

export SOM, cosine_similarity, plasma

def _check_samples(samples):
  unless torch.Tensor === samples and samples.rank == 2 and samples.numel > 0:
    raise ValueError("expected a nonempty [samples, features] tensor")
  unless not samples.sparse? and samples.dtype.name in ["float32", "float64"]:
    raise ValueError("expected a dense float32 or float64 tensor")
  if (samples - samples).ne(0).sum.item > 0:
    raise ValueError("samples must be finite")

def _row_norm_squared(rows): rows.square.sum(dim: 1, keepdim: true)

def _unit_rows(rows):
  magnitudes = rows.abs
  largest = magnitudes.gather(1, magnitudes.argmax(dim: 1, keepdim: true))
  # Scale first so squaring is safe for very large and very small vectors.
  scaled = rows / (largest + largest.eq(0))
  norms = _row_norm_squared(scaled).sqrt
  scaled / (norms + norms.eq(0))

def _cosine_matrix(unit_rows):
  unit_rows.matmul(unit_rows.transpose(0, 1)).clamp(min: -1, max: 1)

def cosine_similarity(samples):
  torch.no_grad:
    _check_samples(samples)
    _cosine_matrix(_unit_rows(samples))

def plasma():
  plot.palette([
    "#0d0887", "#220690", "#310597", "#41049d", "#4e02a2", "#5b01a5",
    "#6700a8", "#7501a8", "#8104a7", "#8d0ba5", "#9814a0", "#a21d9a",
    "#ad2793", "#b6308b", "#bf3984", "#c7427c", "#cf4c74", "#d6556d",
    "#dd5e66", "#e3685f", "#e97257", "#ef7c51", "#f3874a", "#f79143",
    "#fa9c3c", "#fca934", "#fdb52e", "#fdc229", "#fcce25", "#f9dd25",
    "#f5eb27", "#f0f921"
  ])

class SOM:
  attr width
  attr height
  attr metric

  def init(width, height, metric: :euclidean):
    unless Int === width and width > 0 and Int === height and height > 0:
      raise ValueError("map dimensions must be positive integers")
    unless metric in [:euclidean, :cosine]:
      raise ValueError("metric must be :euclidean or :cosine")
    @width = width
    @height = height
    @metric = metric
    @weights = null

  def fit!(samples, epochs: 40, radius: null, seed: 0):
    raise ValueError("epochs must be a positive integer") unless Int === epochs and epochs > 0
    radius = Math.max(1.0, Math.max(@width, @height) / 4.0) if radius == null
    unless radius >= 0.5 and radius < 1e100:
      raise ValueError("radius must be finite and at least 0.5")
    @weights = torch.no_grad:
      _check_samples(samples)
      data = samples
      data = _unit_rows(data) if @metric == :cosine
      rows = data.shape[0]
      order = torch.Generator(seed:).randperm!(rows).to_a
      initial_indices = Array.of(@width * @height) |i|: order[i % rows]
      indices = torch.tensor(initial_indices, dtype: "int64", device: data.device)
      prototypes = data.index_select(0, indices)
      distances = _hex_distances(data)
      epochs.times |epoch|:
        fraction = epoch / Math.max(1, epochs - 1).float
        sigma = radius * (0.5 / radius) ** fraction
        kernel = (distances.square / (-2.0 * sigma * sigma)).exp
        winners = _scores(data, prototypes).argmax(dim: 1)
        influence = kernel.index_select(1, winners)
        sums = influence.matmul(data)
        if @metric == :cosine:
          # Keep the old direction when the weighted vectors cancel.
          updated = _unit_rows(sums)
          prototypes = updated + prototypes * _row_norm_squared(updated).eq(0)
        else:
          mass = influence.sum(dim: 1, keepdim: true)
          empty = mass.eq(0)
          prototypes = sums / (mass + empty) + prototypes * empty
      prototypes
    self

  def weights(): _fitted_weights.copy

  def weights_2d():
    prototypes = _fitted_weights
    prototypes.reshape([@height, @width, prototypes.shape[1]]).copy

  def predict(samples):
    prototypes = _fitted_weights
    torch.no_grad:
      _check_samples(samples)
      unless samples.shape[1] == prototypes.shape[1]:
        raise ValueError("feature count must match the fitted map")
      data = samples.to(device: prototypes.device, dtype: prototypes.dtype)
      data = _unit_rows(data) if @metric == :cosine
      _scores(data, prototypes).argmax(dim: 1)

  def neighbor_similarity():
    prototypes = _fitted_weights
    torch.no_grad:
      neighbors = _hex_distances(prototypes).eq(1)
      counts = neighbors.sum(dim: 1)
      similarities = _cosine_matrix(_unit_rows(prototypes))
      (similarities * neighbors).sum(dim: 1) / (counts + counts.eq(0))

  def hex_heatmap(values, palette: :viridis, label: null, width: 780, height: 540):
    unless Array === values and values.count == @width * @height:
      raise ValueError("expected one value per hexagon in row-major order")
    half_width = Math.sqrt(3.0) / 2.0
    corners = [[0, -1], [half_width, -0.5], [half_width, 0.5],
      [0, 1], [-half_width, 0.5], [-half_width, -0.5]]
    tiles = Array.of(values.count) |i|:
      row = i // @width
      column = i % @width
      x = (2 * column + row % 2) * half_width
      y = row * 1.5
      ring = corners.map |corner|: [x + corner[0], y + corner[1]]
      plot.polygon(ring, value: values[i], id: "#{column}, #{row}")
    plot.figure(width:, height:).axes(show: false)
      .coordinates(aspect: :equal, y: :down)
      .polygon_heatmap(tiles, palette:, label:, stroke: "#e5e7eb")

  def _fitted_weights():
    raise ValueError("fit the map before querying prototypes") if null === @weights
    @weights

  def _scores(data, prototypes):
    products = data.matmul(prototypes.transpose(0, 1))
    return products if @metric == :cosine
    # The data norm is constant across candidates for each row.
    products * 2 - _row_norm_squared(prototypes).transpose(0, 1)

  def _hex_distances(like):
    # Convert the staggered rows to axial coordinates for hex hop distances.
    coordinates = Array.of(@width * @height) |i|:
      row = i // @width
      [i % @width - row // 2, row]
    distances = coordinates.map |a|:
      coordinates.map |b|:
        dq = a[0] - b[0]
        dr = a[1] - b[1]
        Math.max(Math.max(Math.abs(dq), Math.abs(dr)), Math.abs(dq + dr))
    torch.tensor(distances, dtype: like.dtype, device: like.device)
