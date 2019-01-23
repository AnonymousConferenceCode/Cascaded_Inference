def printLogHeader(logFile, val_en = True):
    '''
    Prints a nice header to a log file
    '''
    valStr = 'Val ' if val_en else 'Test'
    logFile.write("|-------||-----------|------------|---------------|----------------|\n")
    logFile.write("| Epoch || " + valStr + " Loss | Train Loss | " + valStr + " Acc. (%) | Train Acc. (%) |\n")
    logFile.write("|-------||-----------|------------|---------------|----------------|\n")

def printLogEntry(logFile, logLine, vl, tl, va, ta):
    '''
    Prints one log entry containing validation loss (vl)
    training loss (tl), validation accuracy (va) and validateion
    accuracy (va)
    '''
    logFile.write('| {0:5d} ||   {1:7.4f} |   {2:8.4f} | {3:13f} | {4:14f} |\n'.format(int(logLine),
                                                                                        float(vl),
                                                                                        float(tl),
                                                                                        float(va),
                                                                                        float(ta)))