from karel import KarelWithCurlyConfigurableParser
from karel.utils import get_rng
from karel.karel import Karel
import random
import copy

from tqdm import tqdm

rng = get_rng(None, 0)


def generate_random_with_distribution(
    distribution,
    no_noops=False,
    stmt_weights=None,
    cond_weights=None,
):
    """
    Return a list of programs whose lengths follow distribution

    distribution is a list, where the ith entry indicates how many
    of length (i+1) should be generated.

    move_weights is a dictionary, with a type of move for keys and
    the associated distribution weight for values.

    cond_weights is a dictionary, where key is conditional type (no_cond, if, ifelse) and value is the associated distribution weight

    state_weights is a dictionary, where key is the state of the board (front_is_clear, left_is_clear, right_is_clear, markers_present, no_markers_present), and valuie is the associated distribution weight

    The programs should be shuffled (i.e., don't generate all the programs of
    one length before moving on to the next one).

    For instance, if distribution = [1, 2, 5], then you should return
    - 1 program of length 1
    - 2 programs of length 2
    - 5 programs of length 3
    """
    distribution = list(distribution)  # don't mutate distribution

    if stmt_weights is None:
        stmt_weights = {
            "pick_marker": 2,
            "put_marker": 2,
            "move": 5,
            "turn_right": 3,
            "turn_left": 3,
            "if": 1,
            "ifelse": 1,
        }

    if cond_weights is None:
        cond_weights = {
            "left_is_clear": 1,
            "right_is_clear": 1,
            "front_is_clear": 1,
            "markers_present": 1,
            "no_markers_present": 1,
        }

    cond_prods = list(cond_weights.keys())
    stmt_prods = list(stmt_weights.keys())

    def choose_prod(prods, weights_dict):
        return random.choices(prods, weights=weights_dict.values(), k=1)[0]

    def execute_token(state, token: str):
        return getattr(state, token)()

    def generate_stmts(length, states):
        if no_noops:
            assert states
        # assert len(states) == 10

        # generate a program of the appropriate length
        stmts = []
        _stmt_weights = dict(stmt_weights)
        stmts_left = length
        while stmts_left:
            _cond_weights = dict(cond_weights)
            assert stmts_left > 0
            # check what kind of conditional is possible, given length
            # set ifelse and if weights to 0 if there aren't enough states left
            if stmts_left < 3 or len(states) < 2:
                _stmt_weights["ifelse"] = 0
            if stmts_left < 2 or len(states) < 2:
                _stmt_weights["if"] = 0

            # generate random statement
            token = choose_prod(stmt_prods, _stmt_weights)
            stmts_left -= 1

            # if no_noops, only sample conditionals that have branch coverage
            if token.startswith("if") and no_noops:  # if, ifelse
                valid_conditional_exists = False
                for cond in cond_prods:
                    # state for state in states if evaluate_conditional[cond](state)
                    true_states = [
                        state for state in states if execute_token(state, cond)
                    ]
                    if len(true_states) % len(states) == 0:
                        _cond_weights[cond] = 0
                    else:
                        valid_conditional_exists = True

                if not valid_conditional_exists:
                    # resample a non-conditional
                    _stmt_weights["ifelse"] = 0
                    _stmt_weights["if"] = 0
                    token = choose_prod(stmt_prods, _stmt_weights)
                    _stmt_weights["ifelse"] = stmt_weights["ifelse"]
                    _stmt_weights["if"] = stmt_weights["if"]
                    assert not token.startswith("if")

            if token.startswith("if"):
                # generate random conditional
                cond = choose_prod(cond_prods, _cond_weights)
                true_states = [state for state in states if execute_token(state, cond)]
                false_states = [
                    state for state in states if not execute_token(state, cond)
                ]

                # randomly negate the conditional
                if random.random() < 0.5:
                    cond = f"not {cond}"
                    true_states, false_states = false_states, true_states

                if token == "ifelse":
                    length_else_body = (
                        random.choice(range(1, stmts_left - 1)) if stmts_left > 2 else 1
                    )
                    else_branch = generate_stmts(length_else_body, false_states)
                    stmts_left -= length_else_body
                    assert else_branch
                    else_branch = f" else {{ {else_branch} }}"
                else:
                    else_branch = ""

                length_if_body = (
                    random.choice(range(1, stmts_left)) if stmts_left > 1 else 1
                )
                if_branch = generate_stmts(length_if_body, true_states)
                stmts_left -= length_if_body
                assert if_branch

                stmt = f"{token} ( {cond}() ) {{ {if_branch} }}{else_branch}"
                # reset after a conditional is generated
                _stmt_weights = dict(stmt_weights)

            else:
                if no_noops:
                    # limit possible robot moves after this current move
                    if token == "turn_left":
                        _stmt_weights["turn_right"] = 0
                    elif token == "turn_right":
                        _stmt_weights["turn_left"] = 0
                    elif token == "pick_marker":
                        _stmt_weights["put_marker"] = 0
                    elif token == "put_marker":
                        _stmt_weights["pick_marker"] = 0
                    # if the robot moves, then every move becomes viable again
                    else:
                        _stmt_weights = dict(stmt_weights)
                for state in states:
                    # execute_token[token](state)
                    execute_token(state, token)
                stmt = token + "()"

            stmts.append(stmt)

        return "; ".join(stmts)

    parser = KarelWithCurlyConfigurableParser(
        use_loops=False,
        use_conditionals=True,
        use_markers=True,
    )

    def draw_string(parser):
        state = parser.draw(no_print=True)
        assert state is not None
        return "\n".join(state)

    programs = []
    total = sum(distribution)
    for _ in tqdm(range(total)):
        # Pick a random length to generate
        random_len = random.choices(range(len(distribution)), weights=distribution)[0]
        # Decrement the value at sampled index
        distribution[random_len] -= 1
        # Distribution is 1-indexed
        random_len += 1

        record = {}
        np_record = {}

        start_states = make_states(num_states=10)
        for idx in range(len(start_states)):
            record[f"input{idx}"] = karel_to_string(start_states[idx])
            np_record[f"input{idx}"] = start_states[idx].state

        states = copy.deepcopy(start_states)[:5]
        body = generate_stmts(random_len, states)
        program = f"def run() {{ {body} }}"
        for idx in range(len(start_states)):
            parser.karel = start_states[idx]
            parser.run(program)
            end_state = parser.karel
            if idx < len(states):
                assert check_state(states[idx], end_state)
            record[f"output{idx}"] = karel_to_string(end_state)
            np_record[f"output{idx}"] = end_state.state

        programs.append((program, record, np_record))
    assert sum(distribution) == 0
    assert len(programs) == total
    return programs


def check_state(executed_state, end_state):
    return (
        end_state.world == executed_state.world
        and end_state.markers == executed_state.markers
        and end_state.hero.position == executed_state.hero.position
        and end_state.hero.facing == executed_state.hero.facing
    )


def make_states(num_states=10, world_size=(8, 8)):
    return [
        Karel(debug=False, rng=rng, world_size=world_size) for _ in range(num_states)
    ]


def karel_to_string(k):
    state = k.draw(no_print=True)
    return "\n".join(state)


def test_karel():
    state = make_states()[0]
    state.world  # print(state.world)
    state.markers  # print(state.markers)
    state.hero  # print(state.hero)
    karel_to_string(state)  # print(karel_to_string(state))
    print("karel OK")


def test_rng():
    states = make_states()
    s1 = karel_to_string(states[0])

    states = make_states()
    s2 = karel_to_string(states[0])
    assert s1 != s2, "Failed rng test"
    print("rng OK")


def test_actions():
    state = make_states()[0]
    state.move()
    state.turn_left()
    state.turn_right()
    state.put_marker()
    state.pick_marker()
    print("actions OK")


def test_conditionals():
    state = make_states()[0]
    state.front_is_clear()
    state.left_is_clear()
    state.right_is_clear()
    state.markers_present()
    state.no_markers_present()
    print("conditionals OK")


parser = KarelWithCurlyConfigurableParser(
    use_loops=False,
    use_conditionals=True,
    use_markers=True,
)
parser.new_game(world_size=(8, 8))


def test_programs():
    programs = [
        "def run() { pick_marker(); put_marker(); move(); turn_right(); turn_left() }",
        "def run() { if( front_is_clear() ) { put_marker() } }",
        "def run() { ifelse( front_is_clear() ) { put_marker() } else { pick_marker() } }",
        "def run() { ifelse( not front_is_clear() ) { put_marker() } else { pick_marker() } }",
    ]

    def check_program(program):
        try:
            parser.run(program)
        except:
            return False
        return True

    for program in programs:
        assert check_program(program), f"`{program}` Failed..."

    print("programs OK")


def test_generate():
    distribution = [1000] * 10
    stmt_weights = {
        "pick_marker": 10,
        "put_marker": 20,
        "move": 30,
        "turn_right": 25,
        "turn_left": 25,
        "if": 25,
        "ifelse": 25,
    }
    cond_weights = {
        "front_is_clear": 10,
        "left_is_clear": 20,
        "right_is_clear": 5,
        "markers_present": 15,
        "no_markers_present": 50,
    }

    programs = generate_random_with_distribution(
        distribution,
        True,
        stmt_weights,
        cond_weights,
    )

    counts = [0] * len(distribution)
    for prog, _, _ in programs:
        counts[prog.count("()") - 1 - 1] += 1

    if not counts == distribution:
        raise ValueError(
            f"Wrong distribution generated, expected {distribution} but got {counts}."
        )

    print("generate OK")


if __name__ == "__main__":
    test_karel()
    test_rng()
    test_actions()
    test_conditionals()
    test_programs()
    test_generate()

    print("PASSED!!!")
