def translation_prediction(model_file):

    y = T.ivector('y')
    index = T.lscalar()
    dataset = 'mnist.pkl.gz'
    datasets = loaddata_mnist(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.reshape((10000, 1, 28, 28))

    printimage(test_set_x[15])
    #numpy.save('test_set_label.npy', test_set_y.eval())
    #print('end')
    #return
    #test_set_x = T.Rebroadcast((1, True))(test_set_x)
    #test_set_x = T.Rebroadcast((1, True))(test_set_x)

#    with open('Con_MLP_Train_Trans_0_4.pkl', 'rb') as f:

    with open(model_file, 'rb') as f:
        layer0, layer1, layer2_input, layer2, layer3 = pickle.load(f)

    error_spectrum = numpy.zeros((21, 21))

    #testy = test_set_y.eval()
    #numpy.save('Test_Set_Label.npy', testy)
    #t_test_set_x = theano_translation(test_set_x, -20, -20, borrow=True)
    #printimage(t_test_set_x[0])

    print('Start Predicting...')
    start_time = timeit.default_timer()
    for horizontal in range(-20, 21, 2):
        temp_time_1 = timeit.default_timer()
        for vertical in range(-20, 21, 2):
            predicted_values = 0
            t_test_set_x = theano_translation(test_set_x, horizontal, vertical, borrow=True)
            predict_model = theano.function(inputs=[index],
                                            outputs=layer3.errors(y),
                                            givens={layer0.input: t_test_set_x[index * 500: (index + 1) * 500],
                                                    y: test_set_y[index * 500: (index + 1) * 500]})
            #print('Horizontal Shift:' + str(horizontal) + '; Vertical Shift:' + str(vertical))
            for batch_value in range(0, 20, 1):
                temp_predicted_values = predict_model(batch_value)
                predicted_values = temp_predicted_values + predicted_values
            predicted_values = predicted_values/20
            #print('Error Rate:' + str(predicted_values))
            error_spectrum[vertical/2 + 10, horizontal/2 + 10] = predicted_values

            #numpy.save(model_file+'_Error_Spectrum.npy', error_spectrum)

            #printimage(t_test_set_x[1])
        temp_time_2 = timeit.default_timer()
        print('This loop ran for %.2fm' % ((temp_time_2 - temp_time_1) / 60.))

    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))

    scipy.io.savemat(model_file+'error_spectrum.mat', mdict={'Error_Spectrum': error_spectrum})

    return error_spectrum

    #workbook = xlsxwriter.Workbook('CMLP_Trans_Full_Data.xlsx')

    #worksheet = workbook.add_worksheet(model_file)

    #for row in range(0, 21):
    #    for column in range(0, 21):
    #       worksheet.write(row, column, error_spectrum[row, column])
    #workbook.close()
