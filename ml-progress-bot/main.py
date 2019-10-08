import os
import yaml
import logging
import traceback
from pprint import pformat
from telegram import ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (Updater, CommandHandler, CallbackQueryHandler,
                          MessageHandler, Filters)

from ckpt_progress import get_progress
from ckpt_learning_curves import latest_ckpt_learning_curves
from confusion_matrix import plot_validation_cm


# enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# load config
with open('config.yaml') as stream:
    config = yaml.load(stream)


def parse_args(x):
    args, kwargs = [], []
    for s in x:
        if '=' in s:
            kwargs.append(s.split('='))
        else:
            args.append(s)
    return args, dict(kwargs)

def get_subdirs(dirpath):
    return sorted([d for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, d))])

def safe_action(f):
    def action(bot, update, args):
        try:
            f(bot, update, args)
        except Exception:
            update.message.reply_text(traceback.format_exc())
    return action

def lc(update, dirpath, args):
    # parse and check params
    args, kwargs = parse_args(args)
    if args:
        dirpath = os.path.join(dirpath, args[0])
    for k in ('b', 'last_epochs'):
        if k in kwargs:
            kwargs[k] = int(kwargs[k])
    for k in ('gamma', 'min_loss', 'max_loss', 'min_acc'):
        if k in kwargs:
            kwargs[k] = float(kwargs[k])

    # plot learning curves
    update.message.reply_text('Plotting learning curves ...')
    val_acc_smoothed = latest_ckpt_learning_curves(dirpath, **kwargs)

    # send the pic
    update.message.reply_photo(open('learning_curves.png', 'rb'))
    update.message.reply_text('Val_acc_smoothed: {0:.4f}'.format(val_acc_smoothed))

def cm(update, dirpath, args):
    args, kwargs = parse_args(args)
    update.message.reply_text('Plotting confusion matrix ...')

    plot_validation_cm(dirpath)
    update.message.reply_photo(open('cm.png', 'rb'))


@safe_action
def status(bot, update, args):
    dirpath = config['default_dir']
    if args:
        dirpath = args[0]
    d = get_progress(dirpath)
    update.message.reply_text(pformat(d), ParseMode.MARKDOWN)

@safe_action
def plot(bot, update, args):
    subdirs = get_subdirs(config['default_dir'])
    keyboard = []
    tmp = []
    for d in subdirs:
        tmp.append(InlineKeyboardButton(d, callback_data=d))
        if len(tmp) == 3:
            keyboard.append(tmp)
            tmp = []
    if tmp: 
        keyboard.append(tmp)
    reply_markup = InlineKeyboardMarkup(keyboard)
    bot.update = update
    bot.args = args
    update.message.reply_text('Choose model:', reply_markup=reply_markup)

def plot_button(bot, update):
    query = update.callback_query
    dirpath = query.data
    bot.edit_message_text(text="Selected model: {}".format(dirpath),
                          chat_id=query.message.chat_id,
                          message_id=query.message.message_id)
    update = bot.update
    args = bot.args
    dirpath = os.path.join(config['default_dir'], dirpath)
    try:
        lc(update, dirpath, args)
        cm(update, dirpath, args)
    except Exception:
        update.message.reply_text(traceback.format_exc())

def start(bot, update):
    print update.message.chat.id


def main():
    # launch bot
    updater = Updater(config['token'])
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('status', status, pass_args=True))
    dp.add_handler(CommandHandler('plot', plot, pass_args=True))
    dp.add_handler(CallbackQueryHandler(plot_button))
    dp.add_handler(MessageHandler(Filters.text, status))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
