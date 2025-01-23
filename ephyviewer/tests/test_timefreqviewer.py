import ephyviewer


from  ephyviewer.tests.testing_tools import make_fake_signals, make_fake_spectrogram


def test_timefreqviewer(interactive=False):
    source = make_fake_signals()


    app = ephyviewer.mkQApp()
    view = ephyviewer.TimeFreqViewer(source=source, name='timefreq')
    #~ view.refresh()

    #~ for c in range(source.nb_channel):
        #~ view.by_channel_params['ch'+str(c), 'visible'] = True

    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)
    win.add_view(view)

    if interactive:
        win.show()
        app.exec()
    else:
        # close thread properly
        win.close()
        
def test_timefreqviewer_fromsource(interactive=False):
    # Generate a fake spectrogram
    spectrogram, frequencies, sample_rate, t_start = make_fake_spectrogram()

    # Create the memory source
    source = ephyviewer.InMemoryTimeFreqSource(
        spectrogram=spectrogram,
        frequencies=frequencies,
        sample_rate=sample_rate,
        t_start=t_start
    )

    app = ephyviewer.mkQApp()
    view = ephyviewer.TimeFreqViewer(source=source, name='timefreq_fromsource')
    
    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)
    win.add_view(view)

    if interactive:
        win.show()
        app.exec()
    else:
        # Test in non-interactive mode
        win.close()


if __name__=='__main__':
    test_timefreqviewer(interactive=True)
    test_timefreqviewer_fromsource(interactive=True)
