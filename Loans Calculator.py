print("="*8,"KOPERASI MODELLING", "*"*8)
name = input("Masukkan nama anda: ")
loan = int(input("Masukkan jumlah pinjaman: "))
periode = int(input("Masukkan periode pinjaman (bulan): "))
if periode > 6:
    bunga = 3
    cicilan = True
else:
    bunga = 5
    cicilan = False

bp_bulan = loan/periode
bu_bulan = loan*(bunga/100)

if cicilan:
    print(f'''
    PENGAJUAN SIMPANAN OLEH {name.upper()}
    Metode : Cicilan Bulanan
        Biaya pokok: {round(bp_bulan * periode)}
        Biaya bunga: {round(bu_bulan * periode)}
    Total biaya yang dikeluarkan {(bp_bulan + bu_bulan) * periode}
    cicilan tiap bulan : {(bp_bulan + bu_bulan)}
    
    ''')
else:
    print(f'''
    PENGAJUAN SIMPANAN OLEH {name.upper()}
    Metode : Pembayaran di akhir periode
        Biaya pokok: {round(bp_bulan * periode)}
        Biaya bunga: {round(bu_bulan * periode)}
    Total biaya yang dikeluarkan {((bp_bulan + bu_bulan) * periode)}
    
    ''')
