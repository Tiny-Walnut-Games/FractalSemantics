"""
Fix Unicode characters in Python files to ensure ASCII compatibility.
This script replaces non-ASCII characters with ASCII equivalents.
Usage:
    python fixed_unicode.py input.py output.py
    python fixed_unicode.py --in-place input.py
"""

import os
import re
import sys


def fix_unicode_in_file(file_path):
    """Fix Unicode characters in a single file."""
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
        # Dictionary of Unicode characters to replace with ASCII equivalents
        unicode_replacements = {
            # Superscripts
            '²': '^2',
            '³': '^3',
            '⁴': '^4',
            '⁵': '^5',
            '⁶': '^6',
            '⁷': '^7',
            '⁸': '^8',
            '⁹': '^9',
            '¹': '^1',
            '⁰': '^0',
            # Mathematical symbols
            '∑': 'sum',
            '∏': 'product',
            '∫': 'integral',
            '∂': 'partial',
            '√': 'sqrt',
            '∞': 'infinity',
            '≈': 'approx',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '±': '+/-',
            '÷': '/',
            '×': '*',
            '•': '*',
            '·': '*',
            # Greek letters (common ones)
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'δ': 'delta',
            'ε': 'epsilon',
            'θ': 'theta',
            'λ': 'lambda',
            'μ': 'mu',
            'π': 'pi',
            'σ': 'sigma',
            'τ': 'tau',
            'φ': 'phi',
            'ψ': 'psi',
            'ω': 'omega',
            # Other symbols
            '→': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v',
            '↔': '<->',
            '↕': '<v>',
            '⇒': '=>',
            '⇔': '<=>',
            '∀': 'for_all',
            '∃': 'exists',
            '∈': 'in',
            '∉': 'not_in',
            '⊂': 'subset',
            '⊃': 'superset',
            '⊆': 'subset_eq',
            '⊇': 'superset_eq',
            '∪': 'union',
            '∩': 'intersection',
            '∅': 'empty_set',
            '∇': 'nabla',
            '∆': 'delta',
            # Currency and other symbols
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₹': 'INR',
            '₽': 'RUB',
            '₩': 'KRW',
            '₪': 'ILS',
            '₱': 'PHP',
            '฿': 'THB',
            '₫': 'VND',
            '₴': 'UAH',
            '₸': 'KZT',
            '₺': 'TRY',
            '₼': 'AZN',
            '৲': 'BDT',
            '৳': 'BDT',
            '₮': 'MNT',
            '₰': 'BYN',
            '₿': 'BTC',
            '₨': 'PKR',
            '₢': 'BRL',
            '₣': 'CHF',
            '₧': 'ESP',
            '₳': 'ARS',
            '₵': 'GHS',
            '₶': 'FRF',
            '₷': 'ITL',
            # Emojis and other symbols
            '🎉': 'celebration',
            '❌': 'error',
            '✅': 'success',
            '⚠️': 'warning',
            '🔥': 'fire',
            '⚡': 'lightning',
            '✨': 'sparkles',
            '🚀': 'rocket',
            '💡': 'idea',
            '🔧': 'wrench',
            '⚙️': 'gear',
            '📊': 'chart',
            '📈': 'up_chart',
            '📉': 'down_chart',
            '📋': 'clipboard',
            '📝': 'memo',
            '📄': 'page',
            '📁': 'folder',
            '📂': 'open_folder',
            '🗂': 'card_index_dividers',
            '🗃': 'card_file_box',
            '🗄': 'file_cabinet',
            '📌': 'pushpin',
            '📍': 'round_pushpin',
            '📎': 'paperclip',
            '🖇': 'linked_paperclips',
            '📏': 'straight_ruler',
            '📐': 'triangular_ruler',
            '✂️': 'scissors',
            '🔨': 'hammer',
            '⚒': 'hammer_and_pick',
            '🛠': 'hammer_and_wrench',
            '⛏': 'pick',
            '⛓': 'chains',
            '⚗': 'alembic',
            '🔬': 'microscope',
            '🔭': 'telescope',
            '📡': 'satellite_antenna',
            '💉': 'syringe',
            '💊': 'pill',
            '🚪': 'door',
            '🛏': 'bed',
            '🛋': 'couch_and_lamp',
            '🚽': 'toilet',
            '🚿': 'shower',
            '🛁': 'bathtub',
            '🗿': 'moai',
            '🗽': 'statue_of_liberty',
            '🗼': 'tokyo_tower',
            '🏰': 'european_castle',
            '🏯': 'japanese_castle',
            '🏟': 'stadium',
            '🏛': 'classical_building',
            '🏗': 'building_construction',
            '🏢': 'office_building',
            '🏬': 'department_store',
            '🏣': 'post_office',
            '🏤': 'european_post_office',
            '🏥': 'hospital',
            '🏦': 'bank',
            '🏨': 'hotel',
            '🏪': 'convenience_store',
            '🏫': 'school',
            '🏩': 'love_hotel',
            '🏭': 'factory',
            '💒': 'wedding',
            '⛪': 'church',
            '🕌': 'mosque',
            '🕍': 'synagogue',
            '⛩': 'shinto_shrine',
            '🕋': 'kaaba',
            '⛲': 'fountain',
            '⛺': 'tent',
            '🌁': 'foggy',
            '🌃': 'night_with_stars',
            '🏙': 'cityscape',
            '🌄': 'sunrise_over_mountains',
            '🌅': 'sunrise',
            '🌆': 'city_sunrise',
            '🌇': 'city_sunset',
            '🌉': 'bridge_at_night',
            '♨️': 'hotsprings',
            '🎠': 'carousel_horse',
            '🎡': 'ferris_wheel',
            '🎢': 'roller_coaster',
            '💈': 'barber_pole',
            '🎪': 'circus_tent',
            '🎫': 'ticket',
            '🎬': 'clapper_board',
            '🎭': 'performing_arts',
            '🎮': 'video_game',
            '🎯': 'direct_hit',
            '🎰': 'slot_machine',
            '🎱': '8ball',
            '🎲': 'game_die',
            '🎳': 'bowling',
            '♟': 'chess_pawn',
            '♞': 'chess_knight',
            '♝': 'chess_bishop',
            '♜': 'chess_rook',
            '♛': 'chess_queen',
            '♚': 'chess_king',
            '♠': 'spades',
            '♥': 'hearts',
            '♦': 'diamonds',
            '♣': 'clubs',
            '🃏': 'black_joker',
            '🀄': 'mahjong_red_dragon',
            '🎴': 'flower_playing_cards',
            '🔇': 'muted_speaker',
            '🔈': 'speaker_low_volume',
            '🔉': 'speaker_medium_volume',
            '🔊': 'speaker_high_volume',
            '📢': 'loudspeaker',
            '📣': 'megaphone',
            '📯': 'postal_horn',
            '🔔': 'bell',
            '🔕': 'bell_with_slash',
            '🎼': 'musical_score',
            '🎵': 'musical_note',
            '🎶': 'multiple_musical_notes',
            '🎤': 'microphone',
            '🎧': 'headphone',
            '📻': 'radio',
            '🎷': 'saxophone',
            '🎸': 'guitar',
            '🎹': 'musical_keyboard',
            '🎺': 'trumpet',
            '🎻': 'violin',
            '🥁': 'drum',
            '📱': 'mobile_phone',
            '📲': 'mobile_phone_with_arrow',
            '☎️': 'telephone',
            '📞': 'telephone_receiver',
            '📟': 'pager',
            '📠': 'fax_machine',
            '🔋': 'battery',
            '🔌': 'electric_plug',
            '💻': 'laptop_computer',
            '🖥': 'desktop_computer',
            '🖨': 'printer',
            '⌨️': 'keyboard',
            '🖱': 'computer_mouse',
            '🖲': 'trackball',
            '💽': 'computer_disk',
            '💾': 'floppy_disk',
            '💿': 'cd',
            '📀': 'dvd',
            '🎥': 'movie_camera',
            '🎞': 'film_frames',
            '📽': 'film_projector',
            '📺': 'television',
            '📷': 'camera',
            '📸': 'camera_with_flash',
            '📹': 'video_camera',
            '📼': 'videocassette',
            '🔍': 'left_pointing_magnifying_glass',
            '🔎': 'right_pointing_magnifying_glass',
            '🕯': 'candle',
            '🔦': 'flashlight',
            '🏮': 'red_paper_lantern',
            '📔': 'notebook_with_decorative_cover',
            '📕': 'closed_book',
            '📖': 'open_book',
            '📗': 'green_book',
            '📘': 'blue_book',
            '📙': 'orange_book',
            '📚': 'books',
            '📓': 'notebook',
            '📒': 'ledger',
            '📃': 'page_with_curl',
            '📜': 'scroll',
            '📰': 'newspaper',
            '🗞': 'rolled_up_newspaper',
            '📑': 'bookmark_tabs',
            '🔖': 'bookmark',
            '🏷': 'label',
            '💰': 'money_bag',
            '💴': 'yen_banknote',
            '💵': 'dollar_banknote',
            '💶': 'euro_banknote',
            '💷': 'pound_banknote',
            '💸': 'money_with_wings',
            '💳': 'credit_card',
            '💹': 'chart_increasing_with_yen',
            '💱': 'currency_exchange',
            '💲': 'heavy_dollar_sign',
            '✉️': 'envelope',
            '📧': 'e-mail',
            '📨': 'incoming_envelope',
            '📩': 'envelope_with_arrow',
            '📤': 'outbox_tray',
            '📥': 'inbox_tray',
            '📦': 'package',
            '📫': 'closed_mailbox_with_raised_flag',
            '📪': 'closed_mailbox_with_lowered_flag',
            '📬': 'open_mailbox_with_raised_flag',
            '📭': 'open_mailbox_with_lowered_flag',
            '📮': 'postbox',
            '🗳': 'ballot_box_with_ballot',
            '✏️': 'pencil',
            '✒️': 'black_nib',
            '🖋': 'fountain_pen',
            '🖊': 'pen',
            '💼': 'briefcase',
            '📅': 'calendar',
            '📆': 'tear-off_calendar',
            '🗒': 'spiral_note_pad',
            '🗓': 'spiral_calendar_pad',
            '📇': 'card_index',
            '🗑': 'wastebasket',
            '🔒': 'locked',
            '🔓': 'unlocked',
            '🔏': 'locked_with_pen',
            '🔐': 'locked_with_key',
            '🔑': 'key',
            '🗝': 'old_key',
            '🗡': 'dagger_knife',
            '⚔': 'crossed_swords',
            '🔫': 'pistol',
            '🏹': 'bow_and_arrow',
            '🛡': 'shield',
            '🔩': 'nut_and_bolt',
            '🗜': 'clamp',
            '⚖': 'scales',
            '🔗': 'link',
        }

        # Replace Unicode characters with ASCII equivalents
        for unicode_char, ascii_equiv in unicode_replacements.items():
            content = content.replace(unicode_char, ascii_equiv)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed Unicode characters in {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory_path):
    """Process all Python files in a directory."""
    import glob

    # Find all Python files in the directory
    python_files = []
    for pattern in ['*.py', '**/*.py']:
        python_files.extend(glob.glob(os.path.join(directory_path, pattern), recursive=True))

    # Remove duplicates and filter out this script
    python_files = list(set(python_files))
    python_files = [f for f in python_files if os.path.abspath(f) != os.path.abspath(__file__)]

    if not python_files:
        print(f"No Python files found in {directory_path}")
        return

    print(f"Found {len(python_files)} Python files:")
    for file in python_files:
        print(f"  - {file}")

    # Ask for confirmation
    response = input(f"\nProcess all {len(python_files)} files in-place? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return

    # Process each file
    print(f"\nProcessing {len(python_files)} files...")
    for file_path in python_files:
        print(f"Processing: {file_path}")
        fix_unicode_in_file(file_path)

    print(f"\nCompleted processing {len(python_files)} files.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python fixed_unicode.py input.py [output.py]")
        print("       python fixed_unicode.py --in-place input.py")
        print("       python fixed_unicode.py directory_path")
        sys.exit(1)

    input_arg = sys.argv[1]

    # Check if input is a directory
    if os.path.isdir(input_arg):
        process_directory(input_arg)
        return

    if sys.argv[1] == "--in-place":
        if len(sys.argv) != 3:
            print("Usage: python fixed_unicode.py --in-place input.py")
            sys.exit(1)

        file_path = sys.argv[2]

        # Self-protection: prevent the script from modifying itself
        if os.path.abspath(file_path) == os.path.abspath(__file__):
            print(f"Skipping {file_path} to avoid infinite loop when fixing in-place")
            return

        fix_unicode_in_file(file_path)
    else:
        if len(sys.argv) != 3:
            print("Usage: python fixed_unicode.py input.py output.py")
            sys.exit(1)
        input_path = sys.argv[1]
        output_path = sys.argv[2]

        # Self-protection: prevent the script from modifying itself
        if os.path.abspath(output_path) == os.path.abspath(__file__):
            print(f"Skipping {output_path} to avoid infinite loop when fixing in-place")
            return

        # Copy input to output before fixing
        with open(input_path, encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(infile.read())
        fix_unicode_in_file(output_path)

if __name__ == "__main__":
    main()
