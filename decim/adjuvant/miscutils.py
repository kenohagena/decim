class utils(object):

    def __init__(self):
        self.regressor_names = {
            'C(stimulus, levels=s)[T.vertical]': 'stimulus_vertical',
            'C(stimulus, levels=s)[T.horizontal]': 'stimulus_horizontal',
            'C(response, levels=b)[T.left]': 'response_left',
            'C(response, levels=b)[T.right]': 'response_right',
            'C(rule_resp, levels=r)[T.A]': 'rule_resp_A',
            'C(rule_resp, levels=r)[T.B]': 'rule_resp_B',
            'C(stimulus, levels=s)[T.vertical]:C(rule_resp, levels=r)[T.A]': 'stimulus_vertical_rule_resp_A',
            'C(stimulus, levels=s)[T.horizontal]:C(rule_resp, levels=r)[T.A]': 'stimulus_horizontal_rule_resp_A',
            'C(stimulus, levels=s)[T.vertical]:C(rule_resp, levels=r)[T.B]': 'stimulus_vertical_rule_resp_B',
            'C(stimulus, levels=s)[T.horizontal]:C(rule_resp, levels=r)[T.B]': 'stimulus_horizontal_rule_resp_B',
            'C(response, levels=b)[T.left]:C(rule_resp, levels=r)[T.A]': 'response_left_rule_resp_A',
            'C(response, levels=b)[T.right]:C(rule_resp, levels=r)[T.A]': 'response_right_rule_resp_A',
            'C(response, levels=b)[T.left]:C(rule_resp, levels=r)[T.B]': 'response_left_rule_resp_B',
            'C(response, levels=b)[T.right]:C(rule_resp, levels=r)[T.B]': 'response_right_rule_resp_B',
            'belief': 'belief',
            'np.abs(belief)': 'abs_belief', 'switch': 'switch',
            'np.abs(switch)': 'abs_switch', 'LLR': 'LLR', 'np.abs(LLR)': 'abs_LLR',
            'surprise': 'surprise',
            'C(response_, levels=t)[T.leftA]': 'response_left_rule_resp_A',
            'C(response_, levels=t)[T.leftB]': 'response_left_rule_resp_B',
            'C(response_, levels=t)[T.rightA]': 'response_right_rule_resp_A',
            'C(response_, levels=t)[T.rightB]': 'response_right_rule_resp_B',
            'C(response_, levels=t)[T.missed]': 'response_missed',
            'C(choice_box, levels=t)[T.leftA]': 'response_left_rule_resp_A',
            'C(choice_box, levels=t)[T.leftB]': 'response_left_rule_resp_B',
            'C(choice_box, levels=t)[T.rightA]': 'response_right_rule_resp_A',
            'C(choice_box, levels=t)[T.rightB]': 'response_right_rule_resp_B',
            'C(choice_box, levels=t)[T.missed]': 'response_missed'}
        self.atlase_names = {
            'AAN_DR': 'aan_dr',
            '4th_ventricle': '4th_ventricle',
            'basal_forebrain_4': 'zaborsky_bf4',
            'basal_forebrain_123': 'zaborsky_bf123',
            'LC_Keren_2std': 'keren_lc_2std',
            'LC_standard': 'keren_lc_1std',
            'CIT168': {2: 'NAc', 6: 'SNc', 10: 'VTA'}
        }
