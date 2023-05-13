import itertools
import math

import chess
import chess.pgn

import torch.utils.data as data

FILENAME = "dataset.pgn"


def file_parser(fname: str = FILENAME) -> chess.pgn.Game:
    """
    This function yields one game at a time, taken from the file @fname.

    Args:
    :param fname: the name of the pgn dataset file
    :type fname: str
    :return: one game at a time
    :rtype: chess.pgn.Game
    """
    with open(fname) as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                if game:
                    yield game
                else:
                    return
            except Exception as e:
                print(e)


def game_states(game: chess.pgn.Game) -> tuple[tuple[str, str], str]:
    """
    This function yields one tuple at a time in the form (s_t, s_t+1)

    :param game: the game taken from the database (class chess.pgn.Game)
    :type game: chess.pgn.Game
    :return: the tuple (s_t, s_t+1) and the ground truth (the move that led from s_t to s_t+1)
    :rtype: tuple
    """
    # create a new board
    board = chess.Board()
    # iterate through the moves of @param game
    for i, move in enumerate(game.mainline_moves()):
        # save copy before executing move
        old_board = board.copy(stack=False)
        board.push(move)
        # yield the result one at a time
        yield (old_board, board), move.uci()


class MoveDataset(data.Dataset):

    def __init__(self, fname=FILENAME, max_games=5, board_transform=None, move_transform=None):
        """
        Move Dataset built from pgn file
        :param fname: File path to pgn file
        :param max_games: Maximum number of games to be loaded, set to -1 to load all games
        :param board_transform: function for transforming board state
        :param move_transform: function for transforming move ground truth
        """
        super().__init__()
        self.fname = fname
        self.max_games = max_games
        self.board_transform = board_transform
        self.move_transform = move_transform

        # Get only the first max_games, or all of them if max_games = -1
        it = itertools.islice(file_parser(self.fname), max_games) if max_games != -1 else file_parser(self.fname)
        games = [game for game in it]

        self.states = list(itertools.chain(*(list(game_states(g)) for g in games)))

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        (b1, b2), m = self.states[index]
        if self.board_transform is not None:
            b1, b2 = self.board_transform(b1), self.board_transform(b2)
        if self.move_transform is not None:
            m = self.move_transform(m)
        return (b1, b2), m


def get_dataloader(fname, max_games=5, batch_size=32, num_workers=5,
                   board_transform=None, move_transform=None,
                   split_perc=(0.7, 0.1, 0.2)):
    """
    Get dataloader for move dataset
    :param fname: File path to pgn file
    :param max_games: Maximum number of games to be loaded, set to -1 to load all games
    :param batch_size: batch size
    :param num_workers: workers for the dataloader
    :param board_transform: function for transforming board state
    :param move_transform: function for transforming move ground truth
    :param split_perc: train eval test percentage split, pass a tuple of 3 floats
    :return: train_dataloader, val_dataloader, test_dataloader
    """
    assert sum(split_perc) == 1.0
    dataset = MoveDataset(fname, max_games, board_transform=board_transform, move_transform=move_transform)
    tot_samples = len(dataset)
    t1, t2, t3 = split_perc[0] * tot_samples, (split_perc[0] + split_perc[1]) * tot_samples, tot_samples
    train_idx = range(0, int(t1))
    val_idx = range(int(t1), int(t2))
    test_idx = range(int(t2), int(t3))
    train_dataset = data.Subset(dataset, indices=train_idx)
    val_dataset = data.Subset(dataset, indices=val_idx)
    test_dataset = data.Subset(dataset, indices=test_idx)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    # main()
    for a in file_parser():
        pass

