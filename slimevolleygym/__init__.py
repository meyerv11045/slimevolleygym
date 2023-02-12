import slimevolleygym.slimevolley
import slimevolleygym.mlp
from slimevolleygym.slimevolley import *
from gym.envs.registration import register


####################
# Reg envs for gym #
####################

register(
    id='SlimeVolleyBaseline-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleyEnv'
)

register(
    id='SlimeVolleyPPOExpert-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleyEnvPPOExpert'
)

register(
    id='SlimeVolleyGAExpert-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleyEnvGAExpert'
)

register(
    id='SlimeVolleyCMAExpert-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleyEnvCMAExpert'
)

register(
    id='SlimeVolleyPixel-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleyPixelEnv'
)

register(
    id='SlimeVolleyNoFrameskip-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleyAtariEnv'
)

register(
    id='SlimeVolleySurvivalNoFrameskip-v0',
    entry_point='slimevolleygym.slimevolley:SlimeVolleySurvivalAtariEnv'
)