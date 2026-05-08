from human_plan.utils.mano.backend import create_mano_model

mano_left = create_mano_model(
  'mano_v1_2/models/MANO_LEFT.pkl',
  is_rhand=False,
  num_pca_comps=15,
)
mano_left.to("cpu")

mano_right = create_mano_model(
  'mano_v1_2/models/MANO_RIGHT.pkl',
  is_rhand=True,
  num_pca_comps=15,
)
mano_right.to("cpu")
