/*
 * Interface wrapper code.
 *
 * Generated by SIP 4.19.3
 *
 * Copyright (C) 2009 Autodesk, Inc. and/or its licensors.
 * All Rights Reserved.
 * 
 * The coded instructions, statements, computer programs, and/or related material 
 * (collectively the "Data") in these files contain unpublished information 
 * proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 * Canada and United States of America federal copyright law and by international 
 * treaties. 
 * 
 * The Data may not be disclosed or distributed to third parties, in whole or in
 * part, without the prior written consent of Autodesk, Inc. ("Autodesk").
 * 
 * THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 * ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 * WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 * BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 * NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 * WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 * OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 * 
 * IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 * OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 * OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 * SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 * OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 * HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 * FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 * ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 * WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 * OR DAMAGE. 
 */

#include "sipAPIfbx.h"




PyDoc_STRVAR(doc_FbxSystemUnit_ConvertScene, "ConvertScene(self, FbxScene, FbxSystemUnit.ConversionOptions = FbxSystemUnit.DefaultConversionOptions)\n"
    "ConvertScene(self, FbxScene, FbxNode, FbxSystemUnit.ConversionOptions = FbxSystemUnit.DefaultConversionOptions)");

extern "C" {static PyObject *meth_FbxSystemUnit_ConvertScene(PyObject *, PyObject *);}
static PyObject *meth_FbxSystemUnit_ConvertScene(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxScene* a0;
        const  ::FbxSystemUnit::ConversionOptions& a1def = FbxSystemUnit::DefaultConversionOptions;
        const  ::FbxSystemUnit::ConversionOptions* a1 = &a1def;
        const  ::FbxSystemUnit *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8|J9", &sipSelf, sipType_FbxSystemUnit, &sipCpp, sipType_FbxScene, &a0, sipType_FbxSystemUnit_ConversionOptions, &a1))
        {
            sipCpp->ConvertScene(a0,*a1);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    {
         ::FbxScene* a0;
         ::FbxNode* a1;
        const  ::FbxSystemUnit::ConversionOptions& a2def = FbxSystemUnit::DefaultConversionOptions;
        const  ::FbxSystemUnit::ConversionOptions* a2 = &a2def;
        const  ::FbxSystemUnit *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8J8|J9", &sipSelf, sipType_FbxSystemUnit, &sipCpp, sipType_FbxScene, &a0, sipType_FbxNode, &a1, sipType_FbxSystemUnit_ConversionOptions, &a2))
        {
            sipCpp->ConvertScene(a0,a1,*a2);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxSystemUnit, sipName_ConvertScene, doc_FbxSystemUnit_ConvertScene);

    return NULL;
}


PyDoc_STRVAR(doc_FbxSystemUnit_ConvertChildren, "ConvertChildren(self, FbxNode, FbxSystemUnit, FbxSystemUnit.ConversionOptions = FbxSystemUnit.DefaultConversionOptions)");

extern "C" {static PyObject *meth_FbxSystemUnit_ConvertChildren(PyObject *, PyObject *);}
static PyObject *meth_FbxSystemUnit_ConvertChildren(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
         ::FbxNode* a0;
        const  ::FbxSystemUnit* a1;
        const  ::FbxSystemUnit::ConversionOptions& a2def = FbxSystemUnit::DefaultConversionOptions;
        const  ::FbxSystemUnit::ConversionOptions* a2 = &a2def;
        const  ::FbxSystemUnit *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ8J9|J9", &sipSelf, sipType_FbxSystemUnit, &sipCpp, sipType_FbxNode, &a0, sipType_FbxSystemUnit, &a1, sipType_FbxSystemUnit_ConversionOptions, &a2))
        {
            sipCpp->ConvertChildren(a0,*a1,*a2);

            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxSystemUnit, sipName_ConvertChildren, doc_FbxSystemUnit_ConvertChildren);

    return NULL;
}


PyDoc_STRVAR(doc_FbxSystemUnit_GetScaleFactor, "GetScaleFactor(self) -> float");

extern "C" {static PyObject *meth_FbxSystemUnit_GetScaleFactor(PyObject *, PyObject *);}
static PyObject *meth_FbxSystemUnit_GetScaleFactor(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxSystemUnit *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxSystemUnit, &sipCpp))
        {
            double sipRes;

            sipRes = sipCpp->GetScaleFactor();

            return PyFloat_FromDouble(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxSystemUnit, sipName_GetScaleFactor, doc_FbxSystemUnit_GetScaleFactor);

    return NULL;
}


PyDoc_STRVAR(doc_FbxSystemUnit_GetScaleFactorAsString, "GetScaleFactorAsString(self, bool = True) -> FbxString");

extern "C" {static PyObject *meth_FbxSystemUnit_GetScaleFactorAsString(PyObject *, PyObject *);}
static PyObject *meth_FbxSystemUnit_GetScaleFactorAsString(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        bool a0 = 1;
        const  ::FbxSystemUnit *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B|b", &sipSelf, sipType_FbxSystemUnit, &sipCpp, &a0))
        {
             ::FbxString*sipRes;

            sipRes = new  ::FbxString(sipCpp->GetScaleFactorAsString(a0));

            return sipConvertFromNewType(sipRes,sipType_FbxString,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxSystemUnit, sipName_GetScaleFactorAsString, doc_FbxSystemUnit_GetScaleFactorAsString);

    return NULL;
}


PyDoc_STRVAR(doc_FbxSystemUnit_GetScaleFactorAsString_Plurial, "GetScaleFactorAsString_Plurial(self) -> FbxString");

extern "C" {static PyObject *meth_FbxSystemUnit_GetScaleFactorAsString_Plurial(PyObject *, PyObject *);}
static PyObject *meth_FbxSystemUnit_GetScaleFactorAsString_Plurial(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxSystemUnit *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxSystemUnit, &sipCpp))
        {
             ::FbxString*sipRes;

            sipRes = new  ::FbxString(sipCpp->GetScaleFactorAsString_Plurial());

            return sipConvertFromNewType(sipRes,sipType_FbxString,NULL);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxSystemUnit, sipName_GetScaleFactorAsString_Plurial, doc_FbxSystemUnit_GetScaleFactorAsString_Plurial);

    return NULL;
}


PyDoc_STRVAR(doc_FbxSystemUnit_GetMultiplier, "GetMultiplier(self) -> float");

extern "C" {static PyObject *meth_FbxSystemUnit_GetMultiplier(PyObject *, PyObject *);}
static PyObject *meth_FbxSystemUnit_GetMultiplier(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxSystemUnit *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "B", &sipSelf, sipType_FbxSystemUnit, &sipCpp))
        {
            double sipRes;

            sipRes = sipCpp->GetMultiplier();

            return PyFloat_FromDouble(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxSystemUnit, sipName_GetMultiplier, doc_FbxSystemUnit_GetMultiplier);

    return NULL;
}


PyDoc_STRVAR(doc_FbxSystemUnit_GetConversionFactorTo, "GetConversionFactorTo(self, FbxSystemUnit) -> float");

extern "C" {static PyObject *meth_FbxSystemUnit_GetConversionFactorTo(PyObject *, PyObject *);}
static PyObject *meth_FbxSystemUnit_GetConversionFactorTo(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxSystemUnit* a0;
        const  ::FbxSystemUnit *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxSystemUnit, &sipCpp, sipType_FbxSystemUnit, &a0))
        {
            double sipRes;

            sipRes = sipCpp->GetConversionFactorTo(*a0);

            return PyFloat_FromDouble(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxSystemUnit, sipName_GetConversionFactorTo, doc_FbxSystemUnit_GetConversionFactorTo);

    return NULL;
}


PyDoc_STRVAR(doc_FbxSystemUnit_GetConversionFactorFrom, "GetConversionFactorFrom(self, FbxSystemUnit) -> float");

extern "C" {static PyObject *meth_FbxSystemUnit_GetConversionFactorFrom(PyObject *, PyObject *);}
static PyObject *meth_FbxSystemUnit_GetConversionFactorFrom(PyObject *sipSelf, PyObject *sipArgs)
{
    PyObject *sipParseErr = NULL;

    {
        const  ::FbxSystemUnit* a0;
        const  ::FbxSystemUnit *sipCpp;

        if (sipParseArgs(&sipParseErr, sipArgs, "BJ9", &sipSelf, sipType_FbxSystemUnit, &sipCpp, sipType_FbxSystemUnit, &a0))
        {
            double sipRes;

            sipRes = sipCpp->GetConversionFactorFrom(*a0);

            return PyFloat_FromDouble(sipRes);
        }
    }

    /* Raise an exception if the arguments couldn't be parsed. */
    sipNoMethod(sipParseErr, sipName_FbxSystemUnit, sipName_GetConversionFactorFrom, doc_FbxSystemUnit_GetConversionFactorFrom);

    return NULL;
}


extern "C" {static PyObject *slot_FbxSystemUnit___ne__(PyObject *,PyObject *);}
static PyObject *slot_FbxSystemUnit___ne__(PyObject *sipSelf,PyObject *sipArg)
{
     ::FbxSystemUnit *sipCpp = reinterpret_cast< ::FbxSystemUnit *>(sipGetCppPtr((sipSimpleWrapper *)sipSelf,sipType_FbxSystemUnit));

    if (!sipCpp)
        return 0;

    PyObject *sipParseErr = NULL;

    {
        const  ::FbxSystemUnit* a0;

        if (sipParseArgs(&sipParseErr, sipArg, "1J9", sipType_FbxSystemUnit, &a0))
        {
            bool sipRes;

            sipRes = sipCpp-> ::FbxSystemUnit::operator!=(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    Py_XDECREF(sipParseErr);

    if (sipParseErr == Py_None)
        return NULL;

    return sipPySlotExtend(&sipModuleAPI_fbx, ne_slot, sipType_FbxSystemUnit, sipSelf, sipArg);
}


extern "C" {static PyObject *slot_FbxSystemUnit___eq__(PyObject *,PyObject *);}
static PyObject *slot_FbxSystemUnit___eq__(PyObject *sipSelf,PyObject *sipArg)
{
     ::FbxSystemUnit *sipCpp = reinterpret_cast< ::FbxSystemUnit *>(sipGetCppPtr((sipSimpleWrapper *)sipSelf,sipType_FbxSystemUnit));

    if (!sipCpp)
        return 0;

    PyObject *sipParseErr = NULL;

    {
        const  ::FbxSystemUnit* a0;

        if (sipParseArgs(&sipParseErr, sipArg, "1J9", sipType_FbxSystemUnit, &a0))
        {
            bool sipRes;

            sipRes = sipCpp-> ::FbxSystemUnit::operator==(*a0);

            return PyBool_FromLong(sipRes);
        }
    }

    Py_XDECREF(sipParseErr);

    if (sipParseErr == Py_None)
        return NULL;

    return sipPySlotExtend(&sipModuleAPI_fbx, eq_slot, sipType_FbxSystemUnit, sipSelf, sipArg);
}


/* Call the instance's destructor. */
extern "C" {static void release_FbxSystemUnit(void *, int);}
static void release_FbxSystemUnit(void *sipCppV, int)
{
    delete reinterpret_cast< ::FbxSystemUnit *>(sipCppV);
}


extern "C" {static void dealloc_FbxSystemUnit(sipSimpleWrapper *);}
static void dealloc_FbxSystemUnit(sipSimpleWrapper *sipSelf)
{
    if (sipIsOwnedByPython(sipSelf))
    {
        release_FbxSystemUnit(sipGetAddress(sipSelf), 0);
    }
}


extern "C" {static void *init_type_FbxSystemUnit(sipSimpleWrapper *, PyObject *, PyObject *, PyObject **, PyObject **, PyObject **);}
static void *init_type_FbxSystemUnit(sipSimpleWrapper *, PyObject *sipArgs, PyObject *sipKwds, PyObject **sipUnused, PyObject **, PyObject **sipParseErr)
{
     ::FbxSystemUnit *sipCpp = 0;

    {
        double a0;
        double a1 = 1;

        if (sipParseKwdArgs(sipParseErr, sipArgs, sipKwds, NULL, sipUnused, "d|d", &a0, &a1))
        {
            sipCpp = new  ::FbxSystemUnit(a0,a1);

            return sipCpp;
        }
    }

    return NULL;
}


/* Define this type's Python slots. */
static sipPySlotDef slots_FbxSystemUnit[] = {
    {(void *)slot_FbxSystemUnit___ne__, ne_slot},
    {(void *)slot_FbxSystemUnit___eq__, eq_slot},
    {0, (sipPySlotType)0}
};


static PyMethodDef methods_FbxSystemUnit[] = {
    {SIP_MLNAME_CAST(sipName_ConvertChildren), meth_FbxSystemUnit_ConvertChildren, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxSystemUnit_ConvertChildren)},
    {SIP_MLNAME_CAST(sipName_ConvertScene), meth_FbxSystemUnit_ConvertScene, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxSystemUnit_ConvertScene)},
    {SIP_MLNAME_CAST(sipName_GetConversionFactorFrom), meth_FbxSystemUnit_GetConversionFactorFrom, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxSystemUnit_GetConversionFactorFrom)},
    {SIP_MLNAME_CAST(sipName_GetConversionFactorTo), meth_FbxSystemUnit_GetConversionFactorTo, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxSystemUnit_GetConversionFactorTo)},
    {SIP_MLNAME_CAST(sipName_GetMultiplier), meth_FbxSystemUnit_GetMultiplier, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxSystemUnit_GetMultiplier)},
    {SIP_MLNAME_CAST(sipName_GetScaleFactor), meth_FbxSystemUnit_GetScaleFactor, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxSystemUnit_GetScaleFactor)},
    {SIP_MLNAME_CAST(sipName_GetScaleFactorAsString), meth_FbxSystemUnit_GetScaleFactorAsString, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxSystemUnit_GetScaleFactorAsString)},
    {SIP_MLNAME_CAST(sipName_GetScaleFactorAsString_Plurial), meth_FbxSystemUnit_GetScaleFactorAsString_Plurial, METH_VARARGS, SIP_MLDOC_CAST(doc_FbxSystemUnit_GetScaleFactorAsString_Plurial)}
};


extern "C" {static PyObject *varget_FbxSystemUnit_DefaultConversionOptions(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_DefaultConversionOptions(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit::ConversionOptions*sipVal;

    sipVal = new  ::FbxSystemUnit::ConversionOptions( ::FbxSystemUnit::DefaultConversionOptions);

    return sipConvertFromNewType(const_cast< ::FbxSystemUnit::ConversionOptions *>(sipVal), sipType_FbxSystemUnit_ConversionOptions, NULL);
}


extern "C" {static PyObject *varget_FbxSystemUnit_Foot(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_Foot(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;

    sipVal = new  ::FbxSystemUnit( ::FbxSystemUnit::Foot);

    return sipConvertFromNewType(const_cast< ::FbxSystemUnit *>(sipVal), sipType_FbxSystemUnit, NULL);
}


extern "C" {static PyObject *varget_FbxSystemUnit_Inch(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_Inch(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;

    sipVal = new  ::FbxSystemUnit( ::FbxSystemUnit::Inch);

    return sipConvertFromNewType(const_cast< ::FbxSystemUnit *>(sipVal), sipType_FbxSystemUnit, NULL);
}


extern "C" {static PyObject *varget_FbxSystemUnit_Mile(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_Mile(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;

    sipVal = new  ::FbxSystemUnit( ::FbxSystemUnit::Mile);

    return sipConvertFromNewType(const_cast< ::FbxSystemUnit *>(sipVal), sipType_FbxSystemUnit, NULL);
}


extern "C" {static PyObject *varget_FbxSystemUnit_Yard(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_Yard(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;

    sipVal = new  ::FbxSystemUnit( ::FbxSystemUnit::Yard);

    return sipConvertFromNewType(const_cast< ::FbxSystemUnit *>(sipVal), sipType_FbxSystemUnit, NULL);
}


extern "C" {static PyObject *varget_FbxSystemUnit_cm(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_cm(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;

    sipVal = new  ::FbxSystemUnit( ::FbxSystemUnit::cm);

    return sipConvertFromNewType(const_cast< ::FbxSystemUnit *>(sipVal), sipType_FbxSystemUnit, NULL);
}


extern "C" {static PyObject *varget_FbxSystemUnit_dm(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_dm(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;

    sipVal = new  ::FbxSystemUnit( ::FbxSystemUnit::dm);

    return sipConvertFromNewType(const_cast< ::FbxSystemUnit *>(sipVal), sipType_FbxSystemUnit, NULL);
}


extern "C" {static PyObject *varget_FbxSystemUnit_km(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_km(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;

    sipVal = new  ::FbxSystemUnit( ::FbxSystemUnit::km);

    return sipConvertFromNewType(const_cast< ::FbxSystemUnit *>(sipVal), sipType_FbxSystemUnit, NULL);
}


extern "C" {static PyObject *varget_FbxSystemUnit_m(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_m(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;

    sipVal = new  ::FbxSystemUnit( ::FbxSystemUnit::m);

    return sipConvertFromNewType(const_cast< ::FbxSystemUnit *>(sipVal), sipType_FbxSystemUnit, NULL);
}


extern "C" {static PyObject *varget_FbxSystemUnit_mm(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_mm(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;

    sipVal = new  ::FbxSystemUnit( ::FbxSystemUnit::mm);

    return sipConvertFromNewType(const_cast< ::FbxSystemUnit *>(sipVal), sipType_FbxSystemUnit, NULL);
}


extern "C" {static PyObject *varget_FbxSystemUnit_sPredefinedUnits(void *, PyObject *, PyObject *);}
static PyObject *varget_FbxSystemUnit_sPredefinedUnits(void *, PyObject *, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;

    sipVal =  ::FbxSystemUnit::sPredefinedUnits;

    return sipConvertFromType(const_cast< ::FbxSystemUnit *>(sipVal), sipType_FbxSystemUnit, NULL);
}


extern "C" {static int varset_FbxSystemUnit_sPredefinedUnits(void *, PyObject *, PyObject *);}
static int varset_FbxSystemUnit_sPredefinedUnits(void *, PyObject *sipPy, PyObject *)
{
    const  ::FbxSystemUnit*sipVal;
    int sipIsErr = 0;

    sipVal = reinterpret_cast< ::FbxSystemUnit *>(sipForceConvertToType(sipPy,sipType_FbxSystemUnit,NULL,0,NULL,&sipIsErr));

    if (sipIsErr)
        return -1;

     ::FbxSystemUnit::sPredefinedUnits = sipVal;

    return 0;
}

sipVariableDef variables_FbxSystemUnit[] = {
    {ClassVariable, sipName_DefaultConversionOptions, (PyMethodDef *)varget_FbxSystemUnit_DefaultConversionOptions, NULL, NULL, NULL},
    {ClassVariable, sipName_Foot, (PyMethodDef *)varget_FbxSystemUnit_Foot, NULL, NULL, NULL},
    {ClassVariable, sipName_Inch, (PyMethodDef *)varget_FbxSystemUnit_Inch, NULL, NULL, NULL},
    {ClassVariable, sipName_Mile, (PyMethodDef *)varget_FbxSystemUnit_Mile, NULL, NULL, NULL},
    {ClassVariable, sipName_Yard, (PyMethodDef *)varget_FbxSystemUnit_Yard, NULL, NULL, NULL},
    {ClassVariable, sipName_cm, (PyMethodDef *)varget_FbxSystemUnit_cm, NULL, NULL, NULL},
    {ClassVariable, sipName_dm, (PyMethodDef *)varget_FbxSystemUnit_dm, NULL, NULL, NULL},
    {ClassVariable, sipName_km, (PyMethodDef *)varget_FbxSystemUnit_km, NULL, NULL, NULL},
    {ClassVariable, sipName_m, (PyMethodDef *)varget_FbxSystemUnit_m, NULL, NULL, NULL},
    {ClassVariable, sipName_mm, (PyMethodDef *)varget_FbxSystemUnit_mm, NULL, NULL, NULL},
    {ClassVariable, sipName_sPredefinedUnits, (PyMethodDef *)varget_FbxSystemUnit_sPredefinedUnits, (PyMethodDef *)varset_FbxSystemUnit_sPredefinedUnits, NULL, NULL},
};

PyDoc_STRVAR(doc_FbxSystemUnit, "\1FbxSystemUnit(float, float = 1)");


sipClassTypeDef sipTypeDef_fbx_FbxSystemUnit = {
    {
        -1,
        0,
        0,
        SIP_TYPE_CLASS,
        sipNameNr_FbxSystemUnit,
        {0},
        0
    },
    {
        sipNameNr_FbxSystemUnit,
        {0, 0, 1},
        8, methods_FbxSystemUnit,
        0, 0,
        11, variables_FbxSystemUnit,
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    },
    doc_FbxSystemUnit,
    -1,
    -1,
    0,
    slots_FbxSystemUnit,
    init_type_FbxSystemUnit,
    0,
    0,
#if PY_MAJOR_VERSION >= 3
    0,
    0,
#else
    0,
    0,
    0,
    0,
#endif
    dealloc_FbxSystemUnit,
    0,
    0,
    0,
    release_FbxSystemUnit,
    0,
    0,
    0,
    0,
    0,
    0,
    0
};
